# Unity NPC roster

The runtime roster contains six NPCs. Use the `Persona name` value exactly in
both `NpcBehaviorClient.npcName` and `NpcDialogueClient.npcName`.

| Persona name | Persona file | Unity model asset | Recommended scene object name |
| --- | --- | --- | --- |
| Asuna | `asuna.json` | `assets/asuna/source/q.fbx` | `Asuna` |
| Lanyan | `lanyan.json` | `assets/Lanyan_Unity/Lanyan.fbx` | `Lanyan` |
| Loen | `loen.json` | `assets/Loen/Loen.fbx` | `Loen` |
| Frederica | `frederica.json` | `assets/miyamoto-frederica/source/Breathing Idle.fbx` | `Frederica` |
| Nicole | `nicole.json` | `assets/Nicole_Unity/Nicole.fbx` | `Nicole` |
| Sanji | `sanji.json` | `assets/sanji-anime-character/source/crowds_30039.fbx` | `Sanji` |

`assets/YeLuoYa_UnityExport/YeLuoYa_Unity.fbx` is kept as a reserve model and
does not belong to the active six-NPC roster. `assets/player.fbx` is the player,
not an NPC persona.

In Unity, rename imported scene objects such as `q` and `Breathing Idle` to the
recommended names. The GameObject name is for scene clarity; the two `npcName`
fields are what the backend uses to locate the persona JSON.
