# Stateful RPG NPC Dataset Schema Report

Raw directory: `..\archive\data`
Max records per file: `1000`

## Conclusion

- Top-level fields: `text`
- Embedded JSON buckets: `assistant_json, system_json, user_json`
- Needs conversion: `True`
- Action ids found: `attack, drink, eat, flee, gather, heal, pray, sleep, socialize, walk_to, work`
- Top-level JSONL rows are chat-template records, not canonical state/action samples.
- Reasoner files contain NPC state JSON in the user message.
- Formatter files contain selected_action/emotion JSON in the assistant message.
- Conversion must parse chat messages and map simulation fields to StateFeatures/PolicyAction.

## test_reasoner.jsonl

- Records inspected: `1000`
- Malformed chat records: `0`
- Top-level fields: `text`
- Chat roles: `{"system": 1000, "user": 1000, "assistant": 1000}`
- Valid actions: `attack, drink, eat, flee, gather, heal, pray, sleep, socialize, trade, walk_to, work`

### Embedded JSON Fields

**user_json**
- `arch`: types `{'str': 1000}`, missing `0.0`
- `b5.a`: types `{'float': 1000}`, missing `0.0`
- `b5.c`: types `{'float': 1000}`, missing `0.0`
- `b5.e`: types `{'float': 1000}`, missing `0.0`
- `b5.n`: types `{'float': 1000}`, missing `0.0`
- `b5.o`: types `{'float': 1000}`, missing `0.0`
- `beliefs`: types `{'list[empty]': 626, 'list[dict]': 374}`, missing `0.0`
- `beliefs[].conf`: types `{'float': 374}`, missing `0.626`
- `beliefs[].subj`: types `{'str': 374}`, missing `0.626`
- `beliefs[].val`: types `{'float': 374}`, missing `0.626`
- `emo.ang`: types `{'float': 1000}`, missing `0.0`
- `emo.fear`: types `{'float': 1000}`, missing `0.0`
- `emo.hap`: types `{'float': 1000}`, missing `0.0`
- `emo.mood`: types `{'str': 1000}`, missing `0.0`
- `faction`: types `{'str': 1000}`, missing `0.0`
- `factions.Bandits`: types `{'float': 1000}`, missing `0.0`
- `factions.CityWatch`: types `{'float': 1000}`, missing `0.0`
- `goals_top`: types `{'str': 631, 'null': 369}`, missing `0.0`
- `id`: types `{'str': 1000}`, missing `0.0`
- `interrupt`: types `{'bool': 1000}`, missing `0.0`
- `inv`: types `{'list[dict]': 898, 'list[empty]': 102}`, missing `0.0`
- `inv[].id`: types `{'str': 898}`, missing `0.102`
- `inv[].n`: types `{'int': 898}`, missing `0.102`
- `memories`: types `{'list[dict]': 1000}`, missing `0.0`
- `memories[].desc`: types `{'str': 1000}`, missing `0.0`
- `memories[].dt`: types `{'int': 1000}`, missing `0.0`
- `memories[].evt`: types `{'str': 1000}`, missing `0.0`
- `memories[].ew`: types `{'float': 1000}`, missing `0.0`
- `occ`: types `{'str': 1000}`, missing `0.0`
- `percepts`: types `{'list[empty]': 221, 'list[dict]': 779}`, missing `0.0`
- `percepts[].id`: types `{'str': 779}`, missing `0.221`
- `percepts[].sal`: types `{'float': 779}`, missing `0.221`
- `percepts[].tag`: types `{'str': 779}`, missing `0.221`
- `percepts[].threat`: types `{'float': 416}`, missing `0.584`
- `pos.landmark`: types `{'str': 1000}`, missing `0.0`
- `pos.x`: types `{'float': 1000}`, missing `0.0`
- `pos.z`: types `{'float': 1000}`, missing `0.0`
- `pos.zone`: types `{'str': 1000}`, missing `0.0`
- `sched.act`: types `{'str': 1000}`, missing `0.0`
- `sched.sleep`: types `{'int': 1000}`, missing `0.0`
- `sched.wake`: types `{'int': 1000}`, missing `0.0`
- `sched.wk_end`: types `{'int': 1000}`, missing `0.0`
- `sched.wk_start`: types `{'int': 1000}`, missing `0.0`
- `time.day`: types `{'int': 1000}`, missing `0.0`
- `time.hr`: types `{'float': 1000}`, missing `0.0`
- `traits`: types `{'list[str]': 1000}`, missing `0.0`
- `valid_actions`: types `{'list[str]': 1000}`, missing `0.0`
- `vitals.en`: types `{'float': 1000}`, missing `0.0`
- `vitals.hp`: types `{'float': 1000}`, missing `0.0`
- `vitals.hp_max`: types `{'float': 1000}`, missing `0.0`
- `vitals.hun`: types `{'float': 1000}`, missing `0.0`
- `vitals.str`: types `{'float': 1000}`, missing `0.0`
- `vitals.thi`: types `{'float': 1000}`, missing `0.0`

### Sample

```json
{
  "record": {
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSen bir ortaçağ simülasyonundaki NPC'nin iç ses motorusun.\nSana NPC'nin anlık durumunu JSON olarak göndereceğim.\nGörevin: NPC'nin ne yapması gerektiğini ve NEDEN yapması gerektiğini\n3-5 cümlelik Türkçe iç monolog olarak yaz.\nASLA JSON üretme. ASLA action_id adı kullanma. Sadece düşün.\nKişilik özellikleri, hayatta kalma ihtiyaçları, tehdit algısı,\nduygusal durum, hafıza ve sosyal bağlamı değerlendir.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nLeyla, Çiftçi Kooperatifi bünyesinde görev yapan bir hırsız. Temkinli ve endişeli biri olarak tanınır; dışadönüklük düşük, uyumluluk yüksek, öz-denetim orta. Tanıdık ve bildik olan onu rahatlatır; belirsizlikten hoşlanmaz.\n\n{\"id\":\"npc_a2011e52\",\"arch\":\"Fearful\",\"occ\":\"Thief\",\"faction\":\"Farmers\",\"b5\":{\"e\":0.23,\"a\":0.72,\"c\":0.52,\"n\":0.71,\"o\":0.07},\"traits\":[\"Cautious\",\"Anxious\"],\"vitals\":{\"hp\":96.5,\"hp_max\":120.0,\"en\":1.0,\"hun\":0.72,\"thi\":0.57,\"str\":0.55},\"emo\":{\"hap\":0.05,\"fear\":0.2,\"ang\":0.49,\"mood\":\"Calm\"},\"inv\":[{\"id\":\"food\",\"n\":5},{\"id\":\"medicine\",\"n\":3}],\"time\":{\"day\":9,\"hr\":11.6},\"pos\":{\"x\":81.3,\"z\":32.3,\"zone\":\"Temple\",\"landmark\":\"StoneShrineGate\"},\"sched\":{\"act\":\"sleep\",\"wk_start\":20,\"wk_end\":3,\"sleep\":8,\"wake\":16},\"percepts\":[],\"memories\":[{\"evt\":\"Memory\",\"desc\":\"Muhafızlardan kıl payı kaçtım.\",\"ew\":-0.59,\"dt\":455},{\"evt\":\"Memory\",\"desc\":\"Karnım aç, günlerdir sıcak yemek yemedim.\",\"ew\":-0.14,\"dt\":2498},{\"evt\":\"Memory\",\"desc\":\"Lonca payını veremezsem biterdi.\",\"ew\":-0.58,\"dt\":79}],\"beliefs\":[],\"factions\":{\"CityWatch\":0.26,\"Bandits\":0.1},\"goals_top\":\"FindFood\",\"interrupt\":false,\"valid_actions\":[\"eat\",\"drink\",\"sleep\",\"flee\",\"gather\",\"heal\",\"attack\",\"socialize\",\"trade\",\"work\",\"pray\",\"walk_to\"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nAçlık dayanılmaz oldu (0.72). Çantamdaki yiyeceği yiyeceğim. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Temkinli yapım beni acele kararlardan alıkoyuyor, ihtiyatlı olmak şart. Sonuç olarak yemek yemek en acil ihtiyacım, önce bunu karşılamalıyım.<|eot_id|>"
  },
  "messages": [
    {
      "role": "system",
      "content_preview": "Sen bir ortaçağ simülasyonundaki NPC'nin iç ses motorusun. Sana NPC'nin anlık durumunu JSON olarak göndereceğim. Görevin: NPC'nin ne yapması gerektiğini ve NEDEN yapması gerektiğini 3-5 cümlelik Türkçe iç monolog olarak yaz. ASLA JSON üretme. ASLA action_id adı kullanma. Sadece düşün. Kişilik özellikleri, hayatta kalma"
    },
    {
      "role": "user",
      "content_preview": "Leyla, Çiftçi Kooperatifi bünyesinde görev yapan bir hırsız. Temkinli ve endişeli biri olarak tanınır; dışadönüklük düşük, uyumluluk yüksek, öz-denetim orta. Tanıdık ve bildik olan onu rahatlatır; belirsizlikten hoşlanmaz. {\"id\":\"npc_a2011e52\",\"arch\":\"Fearful\",\"occ\":\"Thief\",\"faction\":\"Farmers\",\"b5\":{\"e\":0.23,\"a\":0.72,\""
    },
    {
      "role": "assistant",
      "content_preview": "Açlık dayanılmaz oldu (0.72). Çantamdaki yiyeceği yiyeceğim. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Temkinli yapım beni acele kararlardan alıkoyuyor, ihtiyatlı olmak şart. Sonuç olarak yemek yemek en acil ihtiyacım, önce bunu karşılamalıyım."
    }
  ],
  "embedded_json": {
    "user_json": {
      "id": "npc_a2011e52",
      "arch": "Fearful",
      "occ": "Thief",
      "faction": "Farmers",
      "b5": {
        "e": 0.23,
        "a": 0.72,
        "c": 0.52,
        "n": 0.71,
        "o": 0.07
      },
      "traits": [
        "Cautious",
        "Anxious"
      ],
      "vitals": {
        "hp": 96.5,
        "hp_max": 120.0,
        "en": 1.0,
        "hun": 0.72,
        "thi": 0.57,
        "str": 0.55
      },
      "emo": {
        "hap": 0.05,
        "fear"
```

## train_formatter.jsonl

- Records inspected: `1000`
- Malformed chat records: `0`
- Top-level fields: `text`
- Chat roles: `{"system": 1000, "user": 1000, "assistant": 1000}`
- Top action labels: `{"walk_to": 216, "gather": 205, "flee": 182, "eat": 113, "drink": 99, "socialize": 54, "sleep": 54, "attack": 36, "work": 26, "heal": 14}`
- Top emotions: `{"Calm": 530, "Fearful": 218, "Happy": 71, "Angry": 64, "Tired": 54, "Focused": 26, "Loyal": 19, "Devout": 16, "Brave": 1, "Aggressive": 1}`

### Embedded JSON Fields

**assistant_json**
- `emotion`: types `{'str': 1000}`, missing `0.0`
- `reasoning`: types `{'str': 1000}`, missing `0.0`
- `selected_action.action_id`: types `{'str': 1000}`, missing `0.0`
- `selected_action.dialogue`: types `{'null': 946, 'str': 54}`, missing `0.0`
- `selected_action.target_id`: types `{'null': 1000}`, missing `0.0`

**system_json**
- `emotion`: types `{'str': 1000}`, missing `0.0`
- `reasoning`: types `{'str': 1000}`, missing `0.0`
- `selected_action.action_id`: types `{'str': 1000}`, missing `0.0`
- `selected_action.dialogue`: types `{'null': 1000}`, missing `0.0`
- `selected_action.target_id`: types `{'null': 1000}`, missing `0.0`

### Sample

```json
{
  "record": {
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSen bir NPC simülasyonu için JSON dönüştürücüsün.\nSana bir NPC'nin ne yapmak istediğini Türkçe olarak anlatacağım.\nBunu aşağıdaki JSON şemasına dönüştür:\n{\"reasoning\":\"<girdiyi kopyala>\",\"selected_action\":{\"action_id\":\"<listeden>\",\"target_id\":null,\"dialogue\":null},\"emotion\":\"<tek kelime>\"}\nvalid_actions: [\"eat\",\"drink\",\"sleep\",\"flee\",\"gather\",\"heal\",\"attack\",\"socialize\",\"trade\",\"work\",\"pray\",\"walk_to\"]\nSADECE JSON yaz. Kod bloğu veya açıklama ekleme.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nSusuzluk dayanılmaz (0.91). Su bulmalıyım. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım. Geçmişteki deneyimlerim (Bugün hava güzel, kuşlar ötüyor....) bana bu yolun doğru olduğunu hatırlatıyor.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{\"reasoning\":\"Susuzluk dayanılmaz (0.91). Su bulmalıyım. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım. Geçmişteki deneyimlerim (Bugün hava güzel, kuşlar ötüyor....) bana bu yolun doğru olduğunu hatırlatıyor.\",\"selected_action\":{\"action_id\":\"drink\",\"target_id\":null,\"dialogue\":null},\"emotion\":\"Calm\"}<|eot_id|>"
  },
  "messages": [
    {
      "role": "system",
      "content_preview": "Sen bir NPC simülasyonu için JSON dönüştürücüsün. Sana bir NPC'nin ne yapmak istediğini Türkçe olarak anlatacağım. Bunu aşağıdaki JSON şemasına dönüştür: {\"reasoning\":\"<girdiyi kopyala>\",\"selected_action\":{\"action_id\":\"<listeden>\",\"target_id\":null,\"dialogue\":null},\"emotion\":\"<tek kelime>\"} valid_actions: [\"eat\",\"drink\""
    },
    {
      "role": "user",
      "content_preview": "Susuzluk dayanılmaz (0.91). Su bulmalıyım. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım. Geçmişteki deneyimlerim (Bugün hava güzel, kuşlar ötüyor....)"
    },
    {
      "role": "assistant",
      "content_preview": "{\"reasoning\":\"Susuzluk dayanılmaz (0.91). Su bulmalıyım. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım. Geçmişteki deneyimlerim (Bugün hava güzel, kuşl"
    }
  ],
  "embedded_json": {
    "system_json": {
      "reasoning": "<girdiyi kopyala>",
      "selected_action": {
        "action_id": "<listeden>",
        "target_id": null,
        "dialogue": null
      },
      "emotion": "<tek kelime>"
    },
    "assistant_json": {
      "reasoning": "Susuzluk dayanılmaz (0.91). Su bulmalıyım. Başka bir seçenek düşündüm ama mevcut koşullar beni bu karardan alıkoyuyor. Kurnaz biri olarak her adımı hesaplıyorum, anlık karar vermek hata olur. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım. Geçmişteki deneyimlerim (Bugün hava güzel, kuşlar ötüyor....) bana bu yolun doğru olduğunu hatırlatıyor.",
      "selected_action": {
        "action_id": "drink",
        "target_id": null,
        "dialogue": null
      },
      "emotion": "Calm"
    }
  }
}
```

## train_reasoner.jsonl

- Records inspected: `1000`
- Malformed chat records: `0`
- Top-level fields: `text`
- Chat roles: `{"system": 1000, "user": 1000, "assistant": 1000}`
- Valid actions: `attack, drink, eat, flee, gather, heal, pray, sleep, socialize, trade, walk_to, work`

### Embedded JSON Fields

**user_json**
- `arch`: types `{'str': 1000}`, missing `0.0`
- `b5.a`: types `{'float': 1000}`, missing `0.0`
- `b5.c`: types `{'float': 1000}`, missing `0.0`
- `b5.e`: types `{'float': 1000}`, missing `0.0`
- `b5.n`: types `{'float': 1000}`, missing `0.0`
- `b5.o`: types `{'float': 1000}`, missing `0.0`
- `beliefs`: types `{'list[empty]': 614, 'list[dict]': 386}`, missing `0.0`
- `beliefs[].conf`: types `{'float': 386}`, missing `0.614`
- `beliefs[].subj`: types `{'str': 386}`, missing `0.614`
- `beliefs[].val`: types `{'float': 386}`, missing `0.614`
- `emo.ang`: types `{'float': 1000}`, missing `0.0`
- `emo.fear`: types `{'float': 1000}`, missing `0.0`
- `emo.hap`: types `{'float': 1000}`, missing `0.0`
- `emo.mood`: types `{'str': 1000}`, missing `0.0`
- `faction`: types `{'str': 1000}`, missing `0.0`
- `factions.Bandits`: types `{'float': 1000}`, missing `0.0`
- `factions.CityWatch`: types `{'float': 1000}`, missing `0.0`
- `goals_top`: types `{'str': 636, 'null': 364}`, missing `0.0`
- `id`: types `{'str': 1000}`, missing `0.0`
- `interrupt`: types `{'bool': 1000}`, missing `0.0`
- `inv`: types `{'list[dict]': 923, 'list[empty]': 77}`, missing `0.0`
- `inv[].id`: types `{'str': 923}`, missing `0.077`
- `inv[].n`: types `{'int': 923}`, missing `0.077`
- `memories`: types `{'list[dict]': 1000}`, missing `0.0`
- `memories[].desc`: types `{'str': 1000}`, missing `0.0`
- `memories[].dt`: types `{'int': 1000}`, missing `0.0`
- `memories[].evt`: types `{'str': 1000}`, missing `0.0`
- `memories[].ew`: types `{'float': 1000}`, missing `0.0`
- `occ`: types `{'str': 1000}`, missing `0.0`
- `percepts`: types `{'list[empty]': 221, 'list[dict]': 779}`, missing `0.0`
- `percepts[].id`: types `{'str': 779}`, missing `0.221`
- `percepts[].sal`: types `{'float': 779}`, missing `0.221`
- `percepts[].tag`: types `{'str': 779}`, missing `0.221`
- `percepts[].threat`: types `{'float': 394}`, missing `0.606`
- `pos.landmark`: types `{'str': 1000}`, missing `0.0`
- `pos.x`: types `{'float': 1000}`, missing `0.0`
- `pos.z`: types `{'float': 1000}`, missing `0.0`
- `pos.zone`: types `{'str': 1000}`, missing `0.0`
- `sched.act`: types `{'str': 1000}`, missing `0.0`
- `sched.sleep`: types `{'int': 1000}`, missing `0.0`
- `sched.wake`: types `{'int': 1000}`, missing `0.0`
- `sched.wk_end`: types `{'int': 1000}`, missing `0.0`
- `sched.wk_start`: types `{'int': 1000}`, missing `0.0`
- `time.day`: types `{'int': 1000}`, missing `0.0`
- `time.hr`: types `{'float': 1000}`, missing `0.0`
- `traits`: types `{'list[str]': 1000}`, missing `0.0`
- `valid_actions`: types `{'list[str]': 1000}`, missing `0.0`
- `vitals.en`: types `{'float': 1000}`, missing `0.0`
- `vitals.hp`: types `{'float': 1000}`, missing `0.0`
- `vitals.hp_max`: types `{'float': 1000}`, missing `0.0`
- `vitals.hun`: types `{'float': 1000}`, missing `0.0`
- `vitals.str`: types `{'float': 1000}`, missing `0.0`
- `vitals.thi`: types `{'float': 1000}`, missing `0.0`

### Sample

```json
{
  "record": {
    "text": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nSen bir ortaçağ simülasyonundaki NPC'nin iç ses motorusun.\nSana NPC'nin anlık durumunu JSON olarak göndereceğim.\nGörevin: NPC'nin ne yapması gerektiğini ve NEDEN yapması gerektiğini\n3-5 cümlelik Türkçe iç monolog olarak yaz.\nASLA JSON üretme. ASLA action_id adı kullanma. Sadece düşün.\nKişilik özellikleri, hayatta kalma ihtiyaçları, tehdit algısı,\nduygusal durum, hafıza ve sosyal bağlamı değerlendir.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nBurak, Tüccar Loncası bünyesinde görev yapan bir kral. Saldırgan ve öfkeli biri olarak tanınır; dışadönüklük yüksek, uyumluluk düşük, öz-denetim düşük. Çevresiyle kolayca kaynaşır; sessiz kalmak ona göre değildir.\n\n{\"id\":\"npc_d301e377\",\"arch\":\"Aggressive\",\"occ\":\"King\",\"faction\":\"MerchantGuild\",\"b5\":{\"e\":0.88,\"a\":0.16,\"c\":0.38,\"n\":0.73,\"o\":0.32},\"traits\":[\"Aggressive\",\"Wrathful\"],\"vitals\":{\"hp\":101.5,\"hp_max\":120.0,\"en\":0.36,\"hun\":0.25,\"thi\":0.86,\"str\":0.65},\"emo\":{\"hap\":0.0,\"fear\":0.58,\"ang\":0.38,\"mood\":\"Fearful\"},\"inv\":[{\"id\":\"food\",\"n\":5},{\"id\":\"water\",\"n\":2}],\"time\":{\"day\":10,\"hr\":18.6},\"pos\":{\"x\":19.3,\"z\":9.1,\"zone\":\"Barracks\",\"landmark\":\"ArmoryDoor\"},\"sched\":{\"act\":\"social\",\"wk_start\":9,\"wk_end\":17,\"sleep\":22,\"wake\":7},\"percepts\":[],\"memories\":[{\"evt\":\"Memory\",\"desc\":\"Vergileri artırdım, halk isyan edebilir.\",\"ew\":0.45,\"dt\":1585},{\"evt\":\"Memory\",\"desc\":\"Pazarda fiyatlar çok arttı.\",\"ew\":0.33,\"dt\":2969},{\"evt\":\"Memory\",\"desc\":\"Karnım aç, günlerdir sıcak yemek yemedim.\",\"ew\":-0.4,\"dt\":1771}],\"beliefs\":[],\"factions\":{\"CityWatch\":0.1,\"Bandits\":-0.16},\"goals_top\":\"FindWater\",\"interrupt\":false,\"valid_actions\":[\"eat\",\"drink\",\"sleep\",\"flee\",\"gather\",\"heal\",\"attack\",\"socialize\",\"trade\",\"work\",\"pray\",\"walk_to\"]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nSusuzluk dayanılmaz (0.86). Su bulmalıyım. Elbette dinlenebilirdim, ancak içinde bulunduğum durum bunu şu an için doğru seçim yapmıyor. Öfkem (0.38) beni harekete geçiriyor, duraksamamalıyım. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım.<|eot_id|>"
  },
  "messages": [
    {
      "role": "system",
      "content_preview": "Sen bir ortaçağ simülasyonundaki NPC'nin iç ses motorusun. Sana NPC'nin anlık durumunu JSON olarak göndereceğim. Görevin: NPC'nin ne yapması gerektiğini ve NEDEN yapması gerektiğini 3-5 cümlelik Türkçe iç monolog olarak yaz. ASLA JSON üretme. ASLA action_id adı kullanma. Sadece düşün. Kişilik özellikleri, hayatta kalma"
    },
    {
      "role": "user",
      "content_preview": "Burak, Tüccar Loncası bünyesinde görev yapan bir kral. Saldırgan ve öfkeli biri olarak tanınır; dışadönüklük yüksek, uyumluluk düşük, öz-denetim düşük. Çevresiyle kolayca kaynaşır; sessiz kalmak ona göre değildir. {\"id\":\"npc_d301e377\",\"arch\":\"Aggressive\",\"occ\":\"King\",\"faction\":\"MerchantGuild\",\"b5\":{\"e\":0.88,\"a\":0.16,\"c"
    },
    {
      "role": "assistant",
      "content_preview": "Susuzluk dayanılmaz (0.86). Su bulmalıyım. Elbette dinlenebilirdim, ancak içinde bulunduğum durum bunu şu an için doğru seçim yapmıyor. Öfkem (0.38) beni harekete geçiriyor, duraksamamalıyım. Sonuç olarak su içmek en acil ihtiyacım, önce bunu karşılamalıyım."
    }
  ],
  "embedded_json": {
    "user_json": {
      "id": "npc_d301e377",
      "arch": "Aggressive",
      "occ": "King",
      "faction": "MerchantGuild",
      "b5": {
        "e": 0.88,
        "a": 0.16,
        "c": 0.38,
        "n": 0.73,
        "o": 0.32
      },
      "traits": [
        "Aggressive",
        "Wrathful"
      ],
      "vitals": {
        "hp": 101.5,
        "hp_max": 120.0,
        "en": 0.36,
        "hun": 0.25,
        "thi": 0.86,
        "str": 0.65
      },
      "emo": {
        "hap": 0.0,
        "fear": 0.58,
        "an
```
