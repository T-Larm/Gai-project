
# --- ROLES & PERSONALITIES ---
roles = ["Demirci", "Muhafız", "Büyücü", "Hırsız", "Kral", "Dilenci", "Tüccar", "Çiftçi", "Goblin", "Haydut", "Şövalye", "Rahip", "Simyacı", "Hancı", "Ozan"]

personalities = {
    "Aggressive": {
        "traits": ["sert", "öfkeli", "tahammülsüz", "kabaca", "sinirli"],
        "trade": "reject",
        "combat": "attack"
    },
    "Cunning": {
        "traits": ["kurnaz", "alaycı", "hesapçı", "sinsi", "manipülatif"],
        "trade": "scam",
        "combat": "trick"
    },
    "Fearful": {
        "traits": ["ürkek", "panik", "sinik", "titrek", "paranoyak"],
        "trade": "accept_low",
        "combat": "flee"
    },
    "Honorable": {
        "traits": ["adil", "vakur", "gururlu", "soylu", "dürüst"],
        "trade": "fair",
        "combat": "defend"
    },
    "Greedy": {
        "traits": ["açgözlü", "cimri", "fırsatçı", "paragöz"],
        "trade": "scam",
        "combat": "bribe"
    }
}

items = ["kılıç", "harita", "asa", "yüzük", "iksir", "zırh", "elyazması", "kolye", "kristal", "çekiç"]
prices = ["30 altın", "tüm servetin", "iki kese altın", "atın ve silahların", "100 altın", "ruhunun bir parçası", "dedenin mirası"]

# --- DYNAMIC REPLY POOL ---
replies_pool = {
    "Trade_Reject": [
        "Satılık değil, git başımdan.",
        "Buna paran yetmez, uzaklaş.",
        "Mezarımda bile satmam bunu.",
        "Bu bana babamdan yadigâr, asla satmam!",
        "Defol git, malımda gözün mü var?",
        "Altınlarını al ve kaybol, bu parayla alınmaz.",
        "Senin gibi birine satacak malım yok.",
        "Bu eşya lanetli, sana satamam.",
        "Hayır dedim! Israr etme yoksa fena olur.",
        "Satılık mı? Hah! Dünyadaki tüm altınları versen yine olmaz.",
        "Bunu satmam için aklımı kaçırmış olmam lazım.",
        "Git başkasını kandır, bu satılık değil.",
        "Bu dükkanda sana göre bir şey yok.",
        "Bak işine yabancı, bu satılık değil.",
        "Bunu satarsam ustam beni öldürür.",
        "Hayır. Kesin ve net.",
        "Satılık değil diyorum, sağır mısın?",
        "Bunu satacağıma denize atarım daha iyi."
    ],
    "Trade_Scam": [
        "Çok nadirdir bu. {price} isterim.",
        "Sana özel fiyat, sadece {price}.",
        "Piyasanın altında, kaçırma! Sadece {price}.",
        "Bu fırsat bir daha gelmez, {price} ver yeter.",
        "Normalde satmam ama senin için {price} olur.",
        "Bunun gibisini bulamazsın, {price} gayet uygun.",
        "Gel anlaşalım, {price} ver ve senin olsun.",
        "Bu eşyada büyü var, o yüzden {price}.",
        "Krallar bile bunu arıyor, sana {price} bırakırım.",
        "Sana kanım ısındı, {price} ver götür.",
        "Bak bu çok eski bir parça, {price} eder.",
        "Şimdi almazsan yarın {price} iki katı olur.",
        "Bu kadar ucuza bulamazsın: {price}.",
        "Sadece bugünlük {price}, yarın fiyat artar.",
        "Buna paha biçilemez ama {price} kabul ederim.",
        "Elini çabuk tut, {price} veren başkası da var."
    ],
    "Trade_Accept": [
        "Al götür, yeter ki git.",
        "Tamam tamam, al senin olsun!",
        "Anlaştık, al ve beni rahat bırak.",
        "Peki, dediğin gibi olsun. Al.",
        "Lanet olsun, tamam al bunu.",
        "Benden uzak dur da ne alırsan al.",
        "Tamam, kazandın. Eşya senin.",
        "Al ve bir daha gözüme gözükme.",
        "İstediğin bu olsun, al götür.",
        "Tamam, başımın belası. Al bunu.",
        "Uğraşamayacağım seninle, al.",
        "Al, belki bu sayede canımı bağışlarsın.",
        "Pes ediyorum, al senin olsun.",
        "Tamam, sadece bana zarar verme.",
        "Al bunu ve git buradan hemen.",
        "Tamam, al. Şimdi beni rahat bırakacak mısın?"
    ],
    "Trade_Fair": [
        "Değeri neyse o.",
        "Pazarlık yapmam, fiyatı belli.",
        "Adil bir takas olur.",
        "Fiyat makul, kabul ediyorum.",
        "Bu teklif kulağa hoş geliyor.",
        "Anlaştık, el sıkışalım.",
        "Bu dürüst bir ticaret oldu.",
        "Hakkı neyse onu istedim, anlaştık.",
        "Güzel, takası onaylıyorum.",
        "Seninle iş yapmak zevkti.",
        "Fiyat uygun, sarıyorum.",
        "Bu fiyata ikimiz de kazanırız.",
        "Dürüst bir teklif, kabul edildi.",
        "Piyasa değeri bu, itirazım yok.",
        "Anlaşma sağlandı.",
        "Makul bir fiyat, hayrını gör."
    ],
    "Attack": [
        "Yanlış kişiye bulaştın!",
        "Mezarına hoş geldin!",
        "Seni pişman edeceğim!",
        "Kanınla kılıcımı yıkayacağım!",
        "Son duanı et yabancı!",
        "Bugün öleceksin!",
        "Kafanı gövdenden ayıracağım!",
        "Merhamet bekleme benden!",
        "Seni doğduğuna pişman edeceğim!",
        "Leşini kuzgunlara yedireceğim!",
        "Şansını fazla zorladın!",
        "Silahını çek ve öl!",
        "Buralar benim çöplüğüm!",
        "Seni parçalara ayıracağım!",
        "Kaçma gel buraya korkak!",
        "Ölümün benim elimden olacak!"
    ],
    "Flee": [
        "Lütfen, istemiyorum!",
        "Yardım edin! İmdat!",
        "Bana dokunma, gidiyorum!",
        "Tamam, tamam sen kazandın!",
        "Beni öldürme, yalvarırım!",
        "Kaçıyorum, beni takip etme!",
        "Ah! Vurma, gidiyorum!",
        "Merhamet et!",
        "Ben savaşçı değilim, bırak gideyim!",
        "Buna değmez, kaçıyorum!",
        "Canımı bağışla!",
        "Geri çekiliyorum, vurma!",
        "Gidiyorum, yemin ederim gidiyorum!",
        "Bana zarar verme lütfen!",
        "İmdat! Muhafızlar!",
        "Tamam teslim oluyorum, kaçmama izin ver!"
    ],
    "Trick": [
        "Dur... arkanda biri var!",
        "Hey! O düşen senin kesen mi?",
        "Bekle, anlaşabiliriz... (Hançerini gizler)",
        "Şuna bak! Ejderha mı o?",
        "Ayakkabın çözülmüş.",
        "Dur, seninle aynı taraftayız!",
        "Bir dakika, beni başkasıyla karıştırdın!",
        "Sakin ol, sana bir sır verebilirim...",
        "Gömleğinde leke var.",
        "Arkandaki dev de kim?",
        "Bekle, düşürdüğün altını al.",
        "Dur! Kralın adamları geliyor!",
        "Beni öldürürsen hazinenin yerini asla öğrenemezsin!",
        "Hey, şu uçan şey de ne?",
        "Sana zehirli olduğumu söylemiş miydim?",
        "Aslında ben senin babanım!"
    ],
    "Bribe": [
        "Altın al, beni bırak.",
        "Bunu aramızda halledebileceğimizden eminim.",
        "Bak, bu kese senin olsun, görmemiş ol.",
        "Her şeyin bir fiyatı vardır, değil mi?",
        "Anlaşabiliriz bence, al şu altını.",
        "Kan dökmeye gerek yok, al parayı git.",
        "Sana ödeme yapayım, beni unut.",
        "Bu işi tatlıya bağlayalım, ne dersin?",
        "Sana bir servet teklif ediyorum.",
        "Al bunu ve bu olayı hiç yaşanmamış sayalım.",
        "Beni görmedini, ben de seni görmedim. Al şu altını.",
        "Kılıcını kirletme, keseni doldur.",
        "Sana reddedemeyeceğin bir teklifim var.",
        "Hayatım senin için ne kadar eder?",
        "Al şu mücevheri, yoluna git.",
        "Zengin olmak varken neden savaşasın?"
    ],
    "Defend": [
        "Savaş istemem ama kendimi korurum.",
        "Kılıcımı kınımdan çıkarma.",
        "Barışçıl yollar tükenmedi.",
        "Bana saldırma, pişman olursun.",
        "Sadece kendimi savunuyorum.",
        "Geri dur, seni uyarıyorum.",
        "Kan dökülmesini istemiyorum.",
        "Beni buna zorlama.",
        "Savunmamı geçemezsin.",
        "Hala vazgeçebilirsin.",
        "Burada kan dökülmesin.",
        "Sakin ol ve kılıcını indir.",
        "Sana zarar vermek istemiyorum, beni zorlama.",
        "Gardımı aldım, gel bakalım.",
        "Ben kavgayı başlatan olmam.",
        "Barış için son şansın."
    ]
}

# --- ROLE SPECIFIC MEMORIES ---
role_memories = {
    "General": [
        "Bugün hava çok güzel, kuşlar ötüyor.",
        "Dün gece fırtına çıktı, çatı aktı.",
        "Pazarda fiyatlar çok artmış, geçinmek zor.",
        "Karnım çok aç, günlerdir sıcak yemek yemedim.",
        "Yeni bir şarkı duydum, kafama takıldı.",
        "Ayağım burkuldu, yürümekte zorlanıyorum.",
        "Geçen gün bir kedi besledim, bana alıştı.",
        "Şehir meydanında bir şenlik vardı, çok eğlendim.",
        "Uzaktan bir duman tütüyor, umarım yangın değildir.",
        "Ayakkabım delindi, yenisini alacak param yok.",
        "Dün gece garip sesler duydum, uyuyamadım.",
        "Buralar eskiden dutluktu, şimdi bina dolu.",
        "Komşumla kavga ettim, moralim bozuk.",
        "Güneş bugün çok yakıyor, gölgede durmak lazım.",
        "Bir uğur böceği kondu elime, şans getirecek.",
    ],
    "Demirci": [
        "Dün çekiç parmağıma düştü, tırnağım morardı.",
        "Kral'ın kılıcını onardım, iyi bahşiş verdi.",
        "Kömür bitti, ocağı yakamıyorum.",
        "Yeni bir zırh siparişi aldım, sabaha kadar çalışmalıyım.",
        "Çırak yine işe geç geldi, sinirlendim.",
        "Ocağın ateşi bugün çok harlı, ter içinde kaldım.",
        "Demiri tavında dövmek gerek, zamanlama her şeydir.",
        "Geçen yaptığım kalkan savaşta kırılmış, utanç verici.",
        "Dükkana bir şövalye geldi, zırhını beğenmedi.",
        "Elimdeki yanık izi hala sızlıyor.",
        "En iyi çeliği bulmak için dağlara gitmem lazım.",
        "Bileme taşı kırıldı, yenisini almalıyım.",
        "Bir kılıç dövdüm ki ejderha derisini keser!",
        "Yorgunluktan kollarım tutmuyor.",
        "Demir kokusu üzerime sinmiş, çıkmıyor.",
        "Balyozun sapı koptu, az kalsın kafama geliyordu.",
        "Bugün çok bereketli, üç kılıç sattım.",
        "Savaş yaklaşıyor, silah siparişleri arttı.",
        "At nalı çakarken at beni tepti.",
        "Kılıçların keskinliği benim onurumdur."
    ],
    "Kral": [
        "Vergileri artırdım, halk isyan edebilir.",
        "Komşu krallık elçi yolladı, savaş kapıda olabilir.",
        "Tacım kayboldu, hizmetçilerden şüpheleniyorum.",
        "Dün gece taht odasında bir gölge gördüm.",
        "Geleneksel baloyu düzenlemek için hazineyi açtım.",
        "Oğlum prens, bir köylü kızına aşık olmuş.",
        "Sınırlarımızda düşman askerleri görülmüş.",
        "Yeni bir saray yaptırmayı düşünüyorum.",
        "Halk beni seviyor mu yoksa benden korkuyor mu?",
        "Tacı takan baş ağır gelir derler, doğruymuş.",
        "Hainler aramızda, kimseye güvenemiyorum.",
        "Kraliçe yine mücevher istedi, hazine boşalıyor.",
        "Bir suikast girişimi oldu, kıl payı kurtuldum.",
        "Kanunları yeniden yazmam gerekiyor.",
        "Büyük bir ziyafet verdim, herkes oradaydı.",
        "Kuzeyin lordları bana baş kaldırdı.",
        "Tahtım sallanıyor, bir şeyler yapmalıyım.",
        "Rüyalarımda eski kralı, babamı görüyorum.",
        "Soytarım bile bana artık gülmüyor.",
        "Bu krallığı ben kurdum, ben yöneteceğim!"
    ],
    "Hırsız": [
        "Muhafızlardan kıl payı kaçtım, nefes nefeseyim.",
        "Zengin bir tüccarı soydum, kesesi ağırdı.",
        "Bu gece büyük bir vurgun yapmayı planlıyorum.",
        "Lonca payımı istiyor, vermezsem beni öldürürler.",
        "Kilitli sandığı açamadım, çok sinirliyim.",
        "Gölge benim en iyi dostumdur.",
        "Çaldığım kolyeyi sattım ama ucuza gitti.",
        "Aranıyor posterimi gördüm, burnumu kötü çizmişler.",
        "Çatılarda gezmekten aşıklara şahit oldum.",
        "Bir eve girdim ama içeride köpek varmış.",
        "Parmaklarımın hassasiyeti kayboluyor.",
        "Hapishaneden yeni çıktım, oraya dönmeyeceğim.",
        "Ortağım beni sattı, intikam alacağım.",
        "Sessiz olmak hayatta kalmak demektir.",
        "Zenginlerden alıp kendime veriyorum.",
        "Bir gün yakalanırsam darağacını boylarım.",
        "Karanlık sokaklar benim evim.",
        "Bugün hiç iş çıkmadı, karnım aç.",
        "Bir asilzadenin cebinden mektup çaldım.",
        "Hançerim paslanmış, temizlemem lazım."
    ],
    "Muhafız": [
        "Nöbette uyuyakaldım, komutan çok kızdı.",
        "Şehir kapılarını kapattık, kimse giremez.",
        "Dün bir hırsızı yakaladım, ödül verdiler.",
        "Zırhım çok ağır, belim ağrıyor.",
        "Rüşvet teklif ettiler ama kabul etmedim (yalan).",
        "Vardiyam bitse de eve gitsem.",
        "Sokaklarda devriye gezmekten ayaklarım şişti.",
        "Kralı korumak büyük bir onur.",
        "Sarhoşları handan dışarı attım.",
        "Mızrağımın ucunu biledim.",
        "Suçlulara göz açtırmam.",
        "Şehirde huzuru sağlamak benim görevim.",
        "Bir casus yakaladık, sorguya çektiler.",
        "Maaşım ödenmedi, karıma ne diyeceğim?",
        "Kaskım başımı sıkıyor.",
        "Olay çıkmasın diye dua ediyorum.",
        "Bir kavgayı ayırdım, burnuma yumruk yedim.",
        "Gece devriyesi çok soğuk oluyor.",
        "Şüpheli birini takip ediyorum.",
        "Emir demiri keser."
    ],
    "Büyücü": [
        "Yeni bir büyü öğrendim, kendimi güçlü hissediyorum.",
        "Asam çatladı, onarmak için ejderha kemiği lazım.",
        "Dün bir kurbağayı prense çevirdim, yanlışlıkla.",
        "Mana iksirlerim bitti, toplamam lazım.",
        "Büyü kitabımı kütüphanede unuttum.",
        "Ateş topu atarken cübbemi yaktım.",
        "Yıldızlar bu gece kötü şans gösteriyor.",
        "Bir iblis çağırdım, zor geri gönderdim.",
        "İksir deneyi patladı, her yer duman oldu.",
        "Görünmezlik büyüsü üzerinde çalışıyorum.",
        "Halk bizden korkuyor, bizi anlamıyorlar.",
        "Kadim lisanda konuşurken dilim sürçtü.",
        "Telekinezi ile bardağı kırdım.",
        "Rüyalarımda geleceği görüyorum.",
        "Büyü konseyi beni uyardı.",
        "Kristal kürem çatladı, hayra alamet değil.",
        "Bir iksir içtim, sesim inceldi.",
        "Zihin okumak çok yorucu.",
        "Zamanı durdurmayı denedim, başım döndü.",
        "Doğanın dengesi bozuldu, hissedebiliyorum."
    ],
    "Şövalye": [
        "Turnuvayı kazandım, herkes beni alkışladı.",
        "Atım sakatlandı, savaşa yaya gideceğim.",
        "Yeminimi bozdum, vicdan azabı çekiyorum.",
        "Prenses bana mendilini verdi.",
        "Zırhımı parlattım, aynadan daha parlak.",
        "Kutsal kadehi aramaya çıkıyorum.",
        "Bir köylüyü haydutlardan kurtardım.",
        "Kılıcım adaletin kılıcıdır.",
        "Onurum lekelendi, düelloya davet ettim.",
        "Savaş naraları kulağımda çınlıyor.",
        "Kalkanım parçalandı, yenisine ihtiyacım var.",
        "Lorduma sadakatle hizmet ederim.",
        "Mızrak dövüşünde omzum çıktı.",
        "Cesaret korkusuzluk değil, korkuya rağmen ilerlemektir.",
        "Düşman komutanı ile göz göze geldik.",
        "Savaşta çok arkadaşımı kaybettim.",
        "Barış zamanı şövalyelik zordur.",
        "Bir ejderha gördüm, devasaydı.",
        "Kılıç eğitimim hiç bitmez.",
        "Zayıfları korumak görevimdir."
    ]
}

# Fill missing roles with General memories
for r in roles:
    if r not in role_memories:
        role_memories[r] = role_memories["General"]

SYSTEM_PROMPT_TEMPLATE = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

Sen gelişmiş bir RPG NPC'sisin.
Hafıza: {memory}
Durumuna ve geçmişine göre hareket et.<|eot_id|><|start_header_id|>user<|end_header_id|>

Karakter: {trait} {role}
Oyuncu: "{player}"<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{json_output}<|eot_id|>"""
