using System;
using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Minimal decoder for the server's audio payloads: base64-encoded
    /// 16-bit PCM mono WAV (backend/audio_utils.py) -> AudioClip.
    /// </summary>
    public static class WavUtility
    {
        public static AudioClip FromBase64Wav(string base64, string clipName)
        {
            try
            {
                return FromWavBytes(Convert.FromBase64String(base64), clipName);
            }
            catch (Exception exc)
            {
                Debug.LogWarning($"[WavUtility] failed to decode audio: {exc.Message}");
                return null;
            }
        }

        public static AudioClip FromWavBytes(byte[] wav, string clipName)
        {
            // Standard RIFF header: sample rate at byte 24, channel count at 22.
            int channels = BitConverter.ToInt16(wav, 22);
            int sampleRate = BitConverter.ToInt32(wav, 24);

            // Find the "data" chunk (usually at byte 36, but not guaranteed).
            int pos = 12;
            while (pos + 8 <= wav.Length)
            {
                string chunkId = System.Text.Encoding.ASCII.GetString(wav, pos, 4);
                int chunkSize = BitConverter.ToInt32(wav, pos + 4);
                if (chunkId == "data")
                {
                    pos += 8;
                    int sampleCount = chunkSize / 2; // 16-bit samples
                    var samples = new float[sampleCount];
                    for (int i = 0; i < sampleCount; i++)
                    {
                        samples[i] = BitConverter.ToInt16(wav, pos + i * 2) / 32768f;
                    }
                    var clip = AudioClip.Create(clipName, sampleCount / channels, channels, sampleRate, false);
                    clip.SetData(samples, 0);
                    return clip;
                }
                pos += 8 + chunkSize;
            }
            Debug.LogWarning("[WavUtility] no data chunk found in WAV");
            return null;
        }
    }
}
