using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace GaiNpc
{
    /// <summary>
    /// Snapshot of one NPC's simulation state, matching the schema expected by
    /// POST /act (backend/behavior/native_features.py). Missing fields default
    /// to 0 server-side, so a minimal state works — fill what your scene has.
    /// JSON is built manually because JsonUtility cannot serialize dictionaries.
    /// </summary>
    public class NpcGameState
    {
        public string occ = "Village Steward";
        public string arch = "Diplomatic";
        public string faction = "";
        public List<string> traits = new List<string>();
        public string goalsTop = "";
        public bool interrupt = false;

        // vitals
        public float hp = 100f, hpMax = 120f, en = 0.8f, hun = 0.2f, thi = 0.2f, str = 0.5f;
        // emotions
        public float hap = 0.2f, fear = 0.1f, ang = 0.1f;
        public string mood = "Calm";
        // big five
        public float b5e = 0.5f, b5a = 0.5f, b5c = 0.5f, b5n = 0.5f, b5o = 0.5f;
        // clock
        public int day = 1;
        public float hour = 12f;
        // schedule
        public string schedAct = "idle";
        public int wkStart = 9, wkEnd = 17, sleepAt = 22, wakeAt = 7;

        public struct Percept { public string id; public string tag; public float threat; public float sal; }
        public List<Percept> percepts = new List<Percept>();

        public struct Item { public string id; public int n; }
        public List<Item> inventory = new List<Item>();

        public string ToJson()
        {
            var sb = new StringBuilder(512);
            sb.Append('{');
            AppendStr(sb, "occ", occ); sb.Append(',');
            AppendStr(sb, "arch", arch); sb.Append(',');
            AppendStr(sb, "faction", faction); sb.Append(',');
            AppendStr(sb, "goals_top", goalsTop); sb.Append(',');
            sb.Append("\"interrupt\":").Append(interrupt ? "true" : "false").Append(',');

            sb.Append("\"traits\":[");
            for (int i = 0; i < traits.Count; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append('"').Append(Escape(traits[i])).Append('"');
            }
            sb.Append("],");

            sb.Append("\"vitals\":{");
            AppendNum(sb, "hp", hp); sb.Append(',');
            AppendNum(sb, "hp_max", hpMax); sb.Append(',');
            AppendNum(sb, "en", en); sb.Append(',');
            AppendNum(sb, "hun", hun); sb.Append(',');
            AppendNum(sb, "thi", thi); sb.Append(',');
            AppendNum(sb, "str", str);
            sb.Append("},");

            sb.Append("\"emo\":{");
            AppendNum(sb, "hap", hap); sb.Append(',');
            AppendNum(sb, "fear", fear); sb.Append(',');
            AppendNum(sb, "ang", ang); sb.Append(',');
            AppendStr(sb, "mood", mood);
            sb.Append("},");

            sb.Append("\"b5\":{");
            AppendNum(sb, "e", b5e); sb.Append(',');
            AppendNum(sb, "a", b5a); sb.Append(',');
            AppendNum(sb, "c", b5c); sb.Append(',');
            AppendNum(sb, "n", b5n); sb.Append(',');
            AppendNum(sb, "o", b5o);
            sb.Append("},");

            sb.Append("\"time\":{");
            AppendNum(sb, "day", day); sb.Append(',');
            AppendNum(sb, "hr", hour);
            sb.Append("},");

            sb.Append("\"sched\":{");
            AppendStr(sb, "act", schedAct); sb.Append(',');
            AppendNum(sb, "wk_start", wkStart); sb.Append(',');
            AppendNum(sb, "wk_end", wkEnd); sb.Append(',');
            AppendNum(sb, "sleep", sleepAt); sb.Append(',');
            AppendNum(sb, "wake", wakeAt);
            sb.Append("},");

            sb.Append("\"percepts\":[");
            for (int i = 0; i < percepts.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var p = percepts[i];
                sb.Append('{');
                AppendStr(sb, "id", p.id); sb.Append(',');
                AppendStr(sb, "tag", p.tag); sb.Append(',');
                AppendNum(sb, "threat", p.threat); sb.Append(',');
                AppendNum(sb, "sal", p.sal);
                sb.Append('}');
            }
            sb.Append("],");

            sb.Append("\"inv\":[");
            for (int i = 0; i < inventory.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var item = inventory[i];
                sb.Append('{');
                AppendStr(sb, "id", item.id); sb.Append(',');
                AppendNum(sb, "n", item.n);
                sb.Append('}');
            }
            sb.Append("],");

            sb.Append("\"factions\":{},\"memories\":[]");
            sb.Append('}');
            return sb.ToString();
        }

        private static void AppendStr(StringBuilder sb, string key, string value)
        {
            sb.Append('"').Append(key).Append("\":\"").Append(Escape(value ?? "")).Append('"');
        }

        private static void AppendNum(StringBuilder sb, string key, float value)
        {
            sb.Append('"').Append(key).Append("\":")
              .Append(value.ToString("0.####", CultureInfo.InvariantCulture));
        }

        private static string Escape(string s)
        {
            return (s ?? "").Replace("\\", "\\\\").Replace("\"", "\\\"")
                .Replace("\b", "\\b").Replace("\f", "\\f")
                .Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        }
    }
}
