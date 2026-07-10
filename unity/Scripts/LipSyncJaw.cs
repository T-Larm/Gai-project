using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Amplitude-driven lip-sync fallback (course plan A): these meshes have
    /// no viseme blendshapes, so we rotate the jaw bone by the voice volume.
    /// If the rig has no jaw bone, nods the head bone slightly instead.
    ///
    /// Jaw is assumed un-animated (absolute pose from cached rest); head is
    /// animator-driven every frame, so the nod is applied relatively.
    /// </summary>
    public class LipSyncJaw : MonoBehaviour
    {
        [Tooltip("Voice AudioSource (bark and/or dialogue reply)")]
        public AudioSource source;
        [Tooltip("Auto-found if empty: first bone whose name contains 'jaw'")]
        public Transform jawBone;
        [Tooltip("Auto-found if empty: first bone whose name contains 'head'")]
        public Transform headBone;
        public float maxJawAngle = 18f;
        public float maxHeadAngle = 4f;
        [Tooltip("Amplitude → angle gain")]
        public float gain = 8f;

        private readonly float[] _samples = new float[256];
        private Quaternion _jawRest;
        private bool _hasJawRest;

        void Start()
        {
            foreach (var t in GetComponentsInChildren<Transform>())
            {
                var n = t.name.ToLowerInvariant();
                if (jawBone == null && n.Contains("jaw")) jawBone = t;
                if (headBone == null && n.Contains("head")) headBone = t;
            }
            if (jawBone != null)
            {
                _jawRest = jawBone.localRotation;
                _hasJawRest = true;
            }
        }

        void LateUpdate()
        {
            if (source == null) return;

            float level = 0f;
            if (source.isPlaying)
            {
                source.GetOutputData(_samples, 0);
                float sum = 0f;
                for (int i = 0; i < _samples.Length; i++) sum += _samples[i] * _samples[i];
                level = Mathf.Clamp01(Mathf.Sqrt(sum / _samples.Length) * gain);
            }

            if (_hasJawRest)
            {
                jawBone.localRotation =
                    _jawRest * Quaternion.AngleAxis(level * maxJawAngle, Vector3.right);
            }
            else if (headBone != null && level > 0f)
            {
                // Head is animated: apply on top of this frame's animator pose.
                headBone.localRotation *=
                    Quaternion.AngleAxis(level * maxHeadAngle, Vector3.right);
            }
        }
    }
}
