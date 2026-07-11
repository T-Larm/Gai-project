using System.Text;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace GaiNpc
{
    /// <summary>
    /// Shared dialogue panel for every NPC. It displays player and NPC text,
    /// sends POST /chat, and enables the client's XTTS voice response.
    /// </summary>
    public class NpcDialogueUI : MonoBehaviour
    {
        [Header("Panel")]
        public GameObject panel;
        public Text speakerNameText;
        public Text conversationText;
        public Text statusText;
        public InputField playerInput;
        public Button sendButton;
        public Button closeButton;

        [Header("NPC Head Bubble")]
        [Tooltip("RectTransform that follows the active NPC. Defaults to Panel.")]
        public RectTransform bubbleRoot;
        [Tooltip("Camera rendering the NPCs. Defaults to Camera.main.")]
        public Camera worldCamera;
        [Tooltip("Used when the model has no Humanoid head bone.")]
        public Vector3 fallbackHeadOffset = new Vector3(0f, 2.2f, 0f);
        public Vector2 screenOffset = new Vector2(0f, 24f);
        public bool keepBubbleOnScreen = true;
        [Min(0f)] public float screenPadding = 16f;
        public bool autoConfigureCompactLayout = true;
        public Vector2 bubbleSize = new Vector2(620f, 280f);

        [Header("Dialogue")]
        public bool enableVoiceReplies = true;
        [Min(1)] public int maxVisibleTurns = 4;

        public bool IsOpen => panel != null && panel.activeSelf;

        private readonly StringBuilder transcript = new StringBuilder();
        private NpcDialogueClient activeClient;
        private Canvas rootCanvas;
        private RectTransform bubbleParent;
        private Transform headTarget;
        private EventTrigger inputEventTrigger;
        private EventTrigger.Entry submitEntry;
        private int visibleTurns;

        private void Awake()
        {
            if (bubbleRoot == null && panel != null)
                bubbleRoot = panel.GetComponent<RectTransform>();
            if (bubbleRoot != null)
            {
                rootCanvas = bubbleRoot.GetComponentInParent<Canvas>();
                bubbleParent = bubbleRoot.parent as RectTransform;
            }
            if (worldCamera == null) worldCamera = Camera.main;
            if (autoConfigureCompactLayout) ConfigureCompactLayout();
            ConfigureInputSubmit();

            if (sendButton != null) sendButton.onClick.AddListener(SendCurrentText);
            if (closeButton != null) closeButton.onClick.AddListener(Close);
            if (panel != null) panel.SetActive(false);
        }

        private void OnDestroy()
        {
            if (sendButton != null) sendButton.onClick.RemoveListener(SendCurrentText);
            if (closeButton != null) closeButton.onClick.RemoveListener(Close);
            if (inputEventTrigger != null && submitEntry != null)
                inputEventTrigger.triggers.Remove(submitEntry);
        }

        private void Update()
        {
            if (!IsOpen) return;

            if (Input.GetKeyDown(KeyCode.Escape))
            {
                Close();
                return;
            }

            if (playerInput != null && playerInput.isFocused &&
                (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter)))
            {
                SendCurrentText();
            }
        }

        private void LateUpdate()
        {
            if (IsOpen) UpdateBubblePosition();
        }

        public void Open(NpcDialogueClient client)
        {
            if (client == null || client.IsBusy) return;

            activeClient = client;
            activeClient.speak = enableVoiceReplies;
            headTarget = FindHeadTarget(activeClient.gameObject);
            transcript.Clear();
            visibleTurns = 0;

            if (speakerNameText != null) speakerNameText.text = activeClient.npcName;
            if (conversationText != null) conversationText.text = "";
            SetStatus("");

            if (panel != null) panel.SetActive(true);
            SetInputEnabled(true);
            FocusInput();
        }

        public void Close()
        {
            if (activeClient != null && activeClient.IsBusy) return;
            if (panel != null) panel.SetActive(false);
            activeClient = null;
            headTarget = null;
        }

        public void SendCurrentText()
        {
            if (activeClient == null || activeClient.IsBusy || playerInput == null) return;

            string playerLine = playerInput.text.Trim();
            if (playerLine.Length == 0) return;

            playerInput.text = "";
            AppendLine("You", playerLine);
            SetStatus("Thinking...");
            SetInputEnabled(false);

            string npcName = activeClient.npcName;
            bool receivedSentence = false;
            activeClient.Send(playerLine, (reply, guardReason) =>
            {
                if (string.IsNullOrWhiteSpace(reply))
                {
                    SetStatus("Unable to reach the dialogue server.");
                }
                else
                {
                    if (!receivedSentence) AppendLine(npcName, reply);
                    SetStatus("");
                }

                if (!string.IsNullOrEmpty(guardReason))
                    Debug.Log($"[{npcName}] dialogue guard: {guardReason}");

                SetInputEnabled(true);
                FocusInput();
            }, sentence =>
            {
                if (string.IsNullOrWhiteSpace(sentence)) return;
                if (!receivedSentence)
                {
                    AppendLine(npcName, sentence);
                    receivedSentence = true;
                }
                else
                {
                    AppendToCurrentLine(sentence);
                }
                SetStatus("Speaking...");
            });
        }

        private void AppendLine(string speaker, string line)
        {
            if (visibleTurns >= maxVisibleTurns)
            {
                transcript.Clear();
                visibleTurns = 0;
            }

            if (transcript.Length > 0) transcript.AppendLine().AppendLine();
            transcript.Append(speaker).Append(": ").Append(line);
            visibleTurns++;

            if (conversationText != null) conversationText.text = transcript.ToString();
        }

        private void SetStatus(string value)
        {
            if (statusText != null) statusText.text = value;
        }

        private void AppendToCurrentLine(string text)
        {
            if (transcript.Length > 0) transcript.Append(' ');
            transcript.Append(text);
            if (conversationText != null) conversationText.text = transcript.ToString();
        }

        private void SetInputEnabled(bool value)
        {
            if (playerInput != null) playerInput.interactable = value;
            if (sendButton != null) sendButton.interactable = value;
        }

        private void FocusInput()
        {
            if (playerInput == null || !playerInput.interactable) return;
            playerInput.Select();
            playerInput.ActivateInputField();
        }

        private Transform FindHeadTarget(GameObject npc)
        {
            Animator animator = npc.GetComponentInChildren<Animator>();
            if (animator != null && animator.isHuman)
            {
                Transform head = animator.GetBoneTransform(HumanBodyBones.Head);
                if (head != null) return head;
            }

            return null;
        }

        private void UpdateBubblePosition()
        {
            if (activeClient == null || bubbleRoot == null || bubbleParent == null) return;

            Camera cameraToUse = worldCamera != null ? worldCamera : Camera.main;
            if (cameraToUse == null) return;

            Vector3 worldPosition = headTarget != null
                ? headTarget.position
                : activeClient.transform.position + fallbackHeadOffset;
            Vector3 screenPosition = cameraToUse.WorldToScreenPoint(worldPosition);

            if (screenPosition.z <= 0f) return;

            Camera canvasCamera = rootCanvas != null && rootCanvas.renderMode != RenderMode.ScreenSpaceOverlay
                ? rootCanvas.worldCamera
                : null;
            Vector2 localPoint;
            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
                    bubbleParent, screenPosition, canvasCamera, out localPoint))
                return;

            localPoint += screenOffset;
            if (keepBubbleOnScreen)
                localPoint = ClampToParent(localPoint);

            bubbleRoot.anchoredPosition = localPoint;
        }

        private Vector2 ClampToParent(Vector2 position)
        {
            Rect parentRect = bubbleParent.rect;
            Rect bubbleRect = bubbleRoot.rect;
            Vector2 pivot = bubbleRoot.pivot;

            float minX = parentRect.xMin + bubbleRect.width * pivot.x + screenPadding;
            float maxX = parentRect.xMax - bubbleRect.width * (1f - pivot.x) - screenPadding;
            float minY = parentRect.yMin + bubbleRect.height * pivot.y + screenPadding;
            float maxY = parentRect.yMax - bubbleRect.height * (1f - pivot.y) - screenPadding;

            if (minX <= maxX) position.x = Mathf.Clamp(position.x, minX, maxX);
            if (minY <= maxY) position.y = Mathf.Clamp(position.y, minY, maxY);
            return position;
        }

        private void ConfigureCompactLayout()
        {
            if (bubbleRoot == null) return;

            bubbleRoot.anchorMin = new Vector2(0.5f, 0.5f);
            bubbleRoot.anchorMax = new Vector2(0.5f, 0.5f);
            bubbleRoot.pivot = new Vector2(0.5f, 0f);
            bubbleRoot.sizeDelta = bubbleSize;

            Image background = bubbleRoot.GetComponent<Image>();
            if (background != null)
                background.color = new Color32(24, 29, 35, 242);

            SetRect(speakerNameText, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(20f, -42f), new Vector2(-64f, -12f));
            SetRect(conversationText, new Vector2(0f, 1f), new Vector2(1f, 1f),
                new Vector2(20f, -184f), new Vector2(-20f, -50f));
            SetRect(statusText, new Vector2(0f, 0f), new Vector2(1f, 0f),
                new Vector2(20f, 64f), new Vector2(-20f, 88f));
            SetRect(playerInput, new Vector2(0f, 0f), new Vector2(1f, 0f),
                new Vector2(20f, 16f), new Vector2(-166f, 58f));
            SetRect(sendButton, new Vector2(1f, 0f), new Vector2(1f, 0f),
                new Vector2(-158f, 16f), new Vector2(-64f, 58f));
            SetRect(closeButton, new Vector2(1f, 1f), new Vector2(1f, 1f),
                new Vector2(-54f, -46f), new Vector2(-12f, -12f));

            StyleText(speakerNameText, 20, FontStyle.Bold, new Color32(126, 231, 180, 255));
            StyleText(conversationText, 17, FontStyle.Normal, new Color32(242, 245, 247, 255));
            StyleText(statusText, 14, FontStyle.Italic, new Color32(171, 181, 191, 255));
            if (conversationText != null)
            {
                conversationText.alignment = TextAnchor.UpperLeft;
                conversationText.horizontalOverflow = HorizontalWrapMode.Wrap;
                conversationText.verticalOverflow = VerticalWrapMode.Truncate;
                conversationText.resizeTextForBestFit = true;
                conversationText.resizeTextMinSize = 13;
                conversationText.resizeTextMaxSize = 17;
            }

            SetButtonLabel(sendButton, "Send");
            SetButtonLabel(closeButton, "X");
            maxVisibleTurns = Mathf.Min(maxVisibleTurns, 2);
        }

        private static void SetRect(Component component, Vector2 anchorMin, Vector2 anchorMax,
            Vector2 offsetMin, Vector2 offsetMax)
        {
            if (component == null) return;
            RectTransform rect = component.GetComponent<RectTransform>();
            if (rect == null) return;
            rect.anchorMin = anchorMin;
            rect.anchorMax = anchorMax;
            rect.offsetMin = offsetMin;
            rect.offsetMax = offsetMax;
        }

        private static void StyleText(Text text, int size, FontStyle style, Color color)
        {
            if (text == null) return;
            text.fontSize = size;
            text.fontStyle = style;
            text.color = color;
        }

        private static void SetButtonLabel(Button button, string label)
        {
            if (button == null) return;
            Text text = button.GetComponentInChildren<Text>();
            if (text == null) return;
            text.text = label;
            text.fontSize = 15;
        }

        private void ConfigureInputSubmit()
        {
            if (playerInput == null) return;

            playerInput.lineType = InputField.LineType.SingleLine;
            inputEventTrigger = playerInput.GetComponent<EventTrigger>();
            if (inputEventTrigger == null)
                inputEventTrigger = playerInput.gameObject.AddComponent<EventTrigger>();
            if (inputEventTrigger.triggers == null)
                inputEventTrigger.triggers = new System.Collections.Generic.List<EventTrigger.Entry>();

            submitEntry = new EventTrigger.Entry { eventID = EventTriggerType.Submit };
            submitEntry.callback.AddListener(_ => SendCurrentText());
            inputEventTrigger.triggers.Add(submitEntry);
        }
    }
}
