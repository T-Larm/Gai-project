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
            if (wav == null || wav.Length < 44)
                throw new ArgumentException("WAV data is missing or too short.");
            if (System.Text.Encoding.ASCII.GetString(wav, 0, 4) != "RIFF" ||
                System.Text.Encoding.ASCII.GetString(wav, 8, 4) != "WAVE")
                throw new ArgumentException("Audio payload is not a RIFF/WAVE file.");

            // Standard RIFF header: sample rate at byte 24, channel count at 22.
            int channels = BitConverter.ToInt16(wav, 22);
            int sampleRate = BitConverter.ToInt32(wav, 24);
            if (channels <= 0 || sampleRate <= 0)
                throw new ArgumentException("WAV header contains an invalid channel count or sample rate.");

            // Find the "data" chunk (usually at byte 36, but not guaranteed).
            int pos = 12;
            while (pos + 8 <= wav.Length)
            {
                string chunkId = System.Text.Encoding.ASCII.GetString(wav, pos, 4);
                int chunkSize = BitConverter.ToInt32(wav, pos + 4);
                if (chunkSize < 0) throw new ArgumentException("WAV chunk has an invalid size.");
                if (chunkId == "data")
                {
                    pos += 8;
                    int availableBytes = Mathf.Min(chunkSize, wav.Length - pos);
                    int sampleCount = availableBytes / 2; // 16-bit samples
                    sampleCount -= sampleCount % channels;
                    if (sampleCount <= 0)
                        throw new ArgumentException("WAV data chunk contains no complete samples.");
                    var samples = new float[sampleCount];
                    for (int i = 0; i < sampleCount; i++)
                    {
                        samples[i] = BitConverter.ToInt16(wav, pos + i * 2) / 32768f;
                    }
                    var clip = AudioClip.Create(clipName, sampleCount / channels, channels, sampleRate, false);
                    clip.SetData(samples, 0);
                    return clip;
                }
                long next = (long)pos + 8L + chunkSize + (chunkSize & 1);
                if (next > wav.Length) break;
                pos = (int)next;
            }
            Debug.LogWarning("[WavUtility] no data chunk found in WAV");
            return null;
        }
    }
}
