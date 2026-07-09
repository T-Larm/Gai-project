using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace GaiNpc
{
    /// <summary>
    /// Dialogue channel: POST /chat for full LLM conversation (three-layer
    /// persona + memory + rule guard). Open this UI when the player initiates
    /// talk or NpcBehaviorClient fires OnShouldTalk.
    /// </summary>
    public class NpcDialogueClient : MonoBehaviour
    {
        [Header("Server")]
        public string serverUrl = "http://127.0.0.1:8000";
        public string npcName = "Aldric";

        [Tooltip("Also synthesize reply audio (XTTS, seconds of extra latency)")]
        public bool speak = false;

        [Tooltip("AudioSource for the reply voice when speak=true")]
        public AudioSource voiceSource;

        [Serializable]
        private class ChatResponse
        {
            public string npc;
            public string reply;
            public GuardInfo guard;
            public string audio_base64;
            public int sample_rate;
        }

        [Serializable]
        private class GuardInfo
        {
            public string reason;
        }

        /// <summary>
        /// Send one player line. onReply(replyText, guardReason) — guardReason
        /// is null for normal turns, "secret_low_trust" / "prompt_injection"
        /// when the rule guard constrained the reply (useful for UI effects,
        /// e.g. the NPC narrows their eyes).
        /// </summary>
        public void Send(string playerText, Action<string, string> onReply)
        {
            StartCoroutine(SendCoroutine(playerText, onReply));
        }

        private IEnumerator SendCoroutine(string playerText, Action<string, string> onReply)
        {
            string body = "{\"npc\":\"" + npcName + "\"," +
                          "\"text\":\"" + Escape(playerText) + "\"," +
                          "\"speak\":" + (speak ? "true" : "false") + "}";

            using (var request = new UnityWebRequest(serverUrl + "/chat", "POST"))
            {
                request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                request.downloadHandler = new DownloadHandlerBuffer();
                request.SetRequestHeader("Content-Type", "application/json");
                yield return request.SendWebRequest();

                if (request.result != UnityWebRequest.Result.Success)
                {
                    Debug.LogWarning($"[NpcDialogueClient] /chat failed: {request.error}");
                    onReply?.Invoke(null, null);
                    yield break;
                }

                var response = JsonUtility.FromJson<ChatResponse>(request.downloadHandler.text);
                string guardReason = null;
                if (response.guard != null && !string.IsNullOrEmpty(response.guard.reason))
                {
                    guardReason = response.guard.reason;
                }
                onReply?.Invoke(response.reply, guardReason);

                if (speak && voiceSource != null && !string.IsNullOrEmpty(response.audio_base64))
                {
                    var clip = WavUtility.FromBase64Wav(response.audio_base64, "reply");
                    if (clip != null)
                    {
                        voiceSource.clip = clip;
                        voiceSource.Play();
                    }
                }
            }
        }

        private static string Escape(string s)
        {
            return (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
                            .Replace("\n", "\\n").Replace("\r", "");
        }
    }
}
