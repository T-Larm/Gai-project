using System;
using UnityEngine;

namespace GaiNpc
{
    /// <summary>
    /// Minimal codec for the server's audio payloads: base64-encoded
    /// 16-bit PCM mono WAV (backend/audio_utils.py) -> AudioClip, and
    /// microphone samples -> WAV bytes for POST /transcribe.
    /// </summary>
    public static class WavUtility
    {
        /// <summary>Float samples [-1,1] -> 16-bit PCM WAV file bytes.</summary>
        public static byte[] ToWavBytes(float[] samples, int channels, int sampleRate)
        {
            int dataSize = samples.Length * 2;
            var wav = new byte[44 + dataSize];

            void WriteAscii(int offset, string s)
            {
                System.Text.Encoding.ASCII.GetBytes(s).CopyTo(wav, offset);
            }
            void WriteInt32(int offset, int v) => BitConverter.GetBytes(v).CopyTo(wav, offset);
            void WriteInt16(int offset, short v) => BitConverter.GetBytes(v).CopyTo(wav, offset);

            WriteAscii(0, "RIFF");
            WriteInt32(4, 36 + dataSize);
            WriteAscii(8, "WAVE");
            WriteAscii(12, "fmt ");
            WriteInt32(16, 16);                              // PCM chunk size
            WriteInt16(20, 1);                               // PCM format
            WriteInt16(22, (short)channels);
            WriteInt32(24, sampleRate);
            WriteInt32(28, sampleRate * channels * 2);       // byte rate
            WriteInt16(32, (short)(channels * 2));           // block align
            WriteInt16(34, 16);                              // bits per sample
            WriteAscii(36, "data");
            WriteInt32(40, dataSize);

            for (int i = 0; i < samples.Length; i++)
            {
                short s = (short)(Mathf.Clamp(samples[i], -1f, 1f) * 32767f);
                BitConverter.GetBytes(s).CopyTo(wav, 44 + i * 2);
            }
            return wav;
        }

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
