using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace GaiNpc
{
    /// <summary>
    /// Screen-space dialogue panel for the dialogue channel (POST /chat).
    /// Opened by NpcTalkTrigger; type and press Enter to send, or hold the
    /// talk key (Left Ctrl) and speak — the recording goes to POST /transcribe
    /// (Whisper) and the recognized text is sent automatically. Esc to close.
    /// While open, the player's control scripts are disabled and the cursor
    /// is released. Guard-constrained replies show a small [guard: ...] tag.
    /// Built and wired by the "GAI NPC → Setup Dialogue UI" menu.
    /// </summary>
    public class DialogueUI : MonoBehaviour
    {
        public static DialogueUI Instance { get; private set; }

        [Header("Wired by GaiNpcDialogueSetup")]
        public GameObject panel;
        public Text nameLabel;
        public Text replyText;
        public Text guardText;
        public Text hintText;
        public InputField input;

        [Header("Voice input")]
        [Tooltip("Hold to record, release to transcribe and send. Left Ctrl " +
                 "so it can't collide with typing letters into the input field.")]
        public KeyCode talkKey = KeyCode.LeftControl;
        [Tooltip("Whisper expects 16 kHz; the backend resamples anything else")]
        public int micSampleRate = 16000;
        public int micMaxSeconds = 30;

        private NpcDialogueClient _npc;
        private MonoBehaviour[] _disabledPlayerScripts = Array.Empty<MonoBehaviour>();
        private bool _waiting;
        private AudioClip _micClip;
        private bool _recording;
        private bool _transcribing;

        public bool IsOpen => panel != null && panel.activeSelf;

        void Awake()
        {
            Instance = this;
            if (panel != null) panel.SetActive(false);
            if (hintText != null) hintText.text = "";
            if (input != null) input.onEndEdit.AddListener(OnSubmit);
        }

        void Update()
        {
            if (!IsOpen) return;
            if (Input.GetKeyDown(KeyCode.Escape))
            {
                Close();
                return;
            }
            if (!_waiting && !_transcribing)
            {
                if (Input.GetKeyDown(talkKey) && !_recording) StartRecording();
                if (Input.GetKeyUp(talkKey) && _recording) StopRecordingAndSend();
            }
        }

        public void ShowHint(string text)
        {
            if (!IsOpen && hintText != null) hintText.text = text;
        }

        public void HideHint()
        {
            if (hintText != null) hintText.text = "";
        }

        public void Open(NpcDialogueClient npc)
        {
            _npc = npc;
            panel.SetActive(true);
            HideHint();
            nameLabel.text = npc.npcName;
            replyText.text = $"(hold {talkKey} and speak, or type and press Enter)";
            guardText.text = "";
            SetPlayerControl(false);
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
            input.text = "";
            input.ActivateInputField();
        }

        public void Close()
        {
            if (_recording)
            {
                Microphone.End(null);
                _recording = false;
            }
            panel.SetActive(false);
            _npc = null;
            SetPlayerControl(true);
            Cursor.lockState = CursorLockMode.Locked;
            Cursor.visible = false;
        }

        /// <summary>onEndEdit listener; only fires the send on Enter.</summary>
        public void OnSubmit(string _)
        {
            if (!IsOpen || _waiting || _npc == null) return;
            if (!Input.GetKeyDown(KeyCode.Return) && !Input.GetKeyDown(KeyCode.KeypadEnter)) return;

            var text = input.text.Trim();
            if (text.Length == 0) return;
            SendText(text);
        }

        private void SendText(string text)
        {
            _waiting = true;
            replyText.text = "...";
            guardText.text = "";
            _npc.Send(text, (reply, guardReason) =>
            {
                _waiting = false;
                replyText.text = reply ?? "(no reply — is the backend running?)";
                if (!string.IsNullOrEmpty(guardReason))
                {
                    guardText.text = $"[guard: {guardReason}]";
                }
                if (IsOpen)
                {
                    input.text = "";
                    input.ActivateInputField();
                }
            });
        }

        private void StartRecording()
        {
            if (Microphone.devices.Length == 0)
            {
                replyText.text = "(no microphone found)";
                return;
            }
            _micClip = Microphone.Start(null, false, micMaxSeconds, micSampleRate);
            _recording = true;
            replyText.text = "(listening... release to send)";
        }

        private void StopRecordingAndSend()
        {
            _recording = false;
            int recordedSamples = Microphone.GetPosition(null);
            Microphone.End(null);
            if (_micClip == null || recordedSamples <= 0)
            {
                replyText.text = "(didn't catch that — hold the key while speaking)";
                return;
            }

            var samples = new float[recordedSamples * _micClip.channels];
            _micClip.GetData(samples, 0);
            var wav = WavUtility.ToWavBytes(samples, _micClip.channels, micSampleRate);
            StartCoroutine(TranscribeAndSend(wav));
        }

        [Serializable]
        private class TranscribeResponse
        {
            public string text;
        }

        private IEnumerator TranscribeAndSend(byte[] wav)
        {
            if (_npc == null) yield break;
            _transcribing = true;
            replyText.text = "(transcribing...)";

            var form = new List<IMultipartFormSection>
            {
                new MultipartFormFileSection("file", wav, "speech.wav", "audio/wav"),
            };
            using (var request = UnityWebRequest.Post(_npc.serverUrl + "/transcribe", form))
            {
                yield return request.SendWebRequest();
                _transcribing = false;

                if (request.result != UnityWebRequest.Result.Success)
                {
                    replyText.text = $"(transcribe failed: {request.error})";
                    yield break;
                }

                var response = JsonUtility.FromJson<TranscribeResponse>(request.downloadHandler.text);
                var text = response != null ? (response.text ?? "").Trim() : "";
                if (text.Length == 0)
                {
                    replyText.text = "(didn't catch that — try again)";
                    yield break;
                }

                if (!IsOpen || _npc == null) yield break;
                input.text = text;   // show the player what was recognized
                SendText(text);
            }
        }

        /// <summary>Disable/enable the player rig's scripts (movement, mouse
        /// look) so typing doesn't move the character.</summary>
        private void SetPlayerControl(bool enable)
        {
            if (!enable)
            {
                var player = FindObjectOfType<CharacterController>();
                if (player == null) return;
                _disabledPlayerScripts = player.GetComponents<MonoBehaviour>();
                foreach (var mb in _disabledPlayerScripts)
                {
                    if (mb != null) mb.enabled = false;
                }
            }
            else
            {
                foreach (var mb in _disabledPlayerScripts)
                {
                    if (mb != null) mb.enabled = true;
                }
                _disabledPlayerScripts = Array.Empty<MonoBehaviour>();
            }
        }
    }
}
