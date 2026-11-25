"""
Müzikten görsel kavram seçimi için kavram sözlüğü (CLAP).

CLAP (laion/larger_clap_music) sesi ve metni aynı uzaya koyar. Son ~10 sn'lik
müziği bu listedeki her kavramla karşılaştırırız; en çok benzeyenler difüzyonun
ne çizeceğini belirler. Liste bilerek geniş ve dengeli tutuldu: hangisinin
seçileceğine model karar verir.

CLAP_TEXT : CLAP'ın müzikle karşılaştırdığı ifade
SD_TEXT   : seçilirse difüzyona giden sahne tarifi
"""

CONCEPTS = [
    # Doğa ve hava
    "rain falling on a window", "a thunderstorm with lightning", "heavy snowfall", "thick fog",
    "a strong wind blowing", "ocean waves on a beach", "a calm lake", "a waterfall", "a river in a valley",
    "a dense forest", "an autumn forest with falling leaves", "a blooming flower field", "a desert with dunes",
    "snowy mountains", "a volcano erupting", "a tropical island", "a frozen arctic landscape",
    "a sunrise over hills", "a sunset over the sea", "a starry night sky", "the northern lights",
    "the full moon", "outer space with galaxies", "a cave with glowing crystals", "underwater coral reef",
    "the deep dark ocean",
    # Şehir ve mekan
    "a city at night with neon lights", "a rainy city street", "a busy downtown crowd", "an empty subway station",
    "a highway at night", "skyscrapers", "an abandoned building", "a small village", "a cozy cabin",
    "a candle-lit room", "an empty bedroom", "a cathedral interior", "an ancient temple", "a castle",
    "a smoky jazz club", "a nightclub with lasers", "a huge stadium concert", "a ballroom",
    "a carnival at night", "a train station", "a lighthouse", "a rooftop at dusk", "a garden",
    "a library", "a futuristic city", "a factory with machines", "a battlefield", "ruins of an old city",
    # İnsanlar ve hareket
    "people dancing", "a couple dancing slowly", "a lonely person walking", "a person crying",
    "friends laughing together", "a child playing", "an old man remembering", "lovers embracing",
    "a crowd jumping", "a runner sprinting", "a soldier marching", "a dreamer floating in the air",
    "a person staring out of a window", "a ballerina", "a street musician", "a choir singing",
    "a drummer", "a pianist at a grand piano", "a guitarist on stage", "a violinist", "a DJ at turntables",
    "a rapper on stage", "a rock band", "an orchestra",
    # Hayvanlar
    "birds flying", "a wolf howling", "horses running", "whales swimming", "butterflies",
    "a cat sleeping", "fireflies at night", "a flock of crows",
    # Nesneler ve semboller
    "fire and flames", "smoke", "broken glass", "a clock ticking", "an old photograph",
    "a letter and a rose", "a burning candle", "balloons", "fireworks", "a mirror", "chains",
    "a heart", "a skull", "a crown", "a car driving fast", "a motorcycle", "a spaceship",
    "a robot", "a ghost", "an angel", "a demon", "a dragon", "a mask", "a door opening",
    "a staircase to the sky", "a bridge", "an hourglass",
    # Atmosfer ve soyut
    "a dreamy haze", "a psychedelic swirl", "glitch and digital noise", "geometric patterns",
    "golden light", "darkness and shadows", "a warm sunny afternoon", "cold blue light",
    "a nostalgic memory", "a childhood memory", "a peaceful meditation", "a heroic journey",
    "a chase scene", "an epic battle", "a celebration", "a wedding", "a funeral", "a romantic evening",
    "a road trip", "a summer party", "a winter night", "spring morning", "a mysterious forest at night",
    "a haunted house", "a sci-fi adventure", "a fairy tale", "a western desert town", "an oriental market",
    "a latin street festival", "an african savanna", "a japanese garden", "an indian temple festival",
    # Destansı / kudretli
    "a mighty warrior", "an army of knights charging", "a king on a throne", "a hero raising a sword",
    "a viking ship in a storm", "gods in the clouds", "a gladiator arena", "a samurai", "titans clashing",
    "a majestic eagle soaring", "a phoenix rising from fire", "a lion roaring", "a fortress on a cliff",
    "a war horse galloping", "banners waving over an army", "a giant statue", "a throne room",
    "a sword and shield", "a fleet of warships", "a mountain peak above the clouds", "a triumphant parade",
    "a warrior queen", "an ancient empire", "a cosmic god",
    # Yas / hüzün / yalnızlık
    "a lament for the dead", "a graveyard at dusk", "a lonely grave with flowers", "a person alone on an empty street",
    "a single rose lying on the ground", "a wilted rose", "a widow dressed in black", "a mourning procession",
    "tears falling", "a broken heart", "an empty chair", "an empty swing in the wind", "a funeral in the rain",
    "a weeping angel statue", "a person sitting alone on a bench", "a lonely streetlight at night",
    "a goodbye on a train platform", "an abandoned house in the rain", "a candle burning out", "a letter never sent",
    "a person holding an old photograph", "a cemetery under snow", "a black crow on a gravestone",
    "a lonely boat on a grey sea", "an empty hospital corridor", "a shattered mirror", "dead leaves on wet pavement",
    "a person hugging their knees", "a silhouette walking away into the fog", "a child alone at a window",
    "a dark empty room", "a wreath of white lilies", "a soldier's grave", "a lonely tree on a hill",
    "rain on an empty playground", "a faded wedding photo",
    # Aşk
    "two hands touching", "a kiss in the rain", "a couple under the stars", "a love letter", "a first dance",
    # Umut
    "light breaking through clouds", "a seedling growing from the ground", "a bird freed from a cage",
    "a path leading toward the sunrise", "a rainbow after the storm",
    # Öfke / isyan
    "a burning city", "a riot in the streets", "a screaming face", "a clenched fist raised high",
    "shattered chains", "a storm destroying everything",
    # Nostalji
    "an old radio", "a vintage car on a country road", "a family dinner", "a summer of childhood",
    "an old cinema", "a polaroid of friends",
    # Mistik
    "a wizard casting a spell", "a forest spirit", "a glowing portal", "a sacred ritual by firelight",
    "a crystal ball", "a moonlit witch",
]

# Ruh hali / üslup sözlüğü: sahnenin NASIL çizileceğini belirler (konu CONCEPTS'ten gelir)
MOODS = [
    ("epic and heroic", "epic heroic fantasy art, dramatic lighting, grand scale"),
    ("majestic and grand", "majestic grand artwork, awe-inspiring, monumental"),
    ("triumphant and victorious", "triumphant victorious scene, glorious light"),
    ("powerful and intense", "powerful intense artwork, dynamic, forceful"),
    ("dark and ominous", "dark ominous artwork, foreboding shadows"),
    ("aggressive and angry", "aggressive furious artwork, harsh red light"),
    ("tense and suspenseful", "tense suspenseful scene, cinematic"),
    ("mysterious", "mysterious enigmatic artwork, mist and shadow"),
    ("eerie and haunting", "eerie haunting artwork, ghostly atmosphere"),
    ("melancholic and sad", "melancholic sad artwork, muted colors"),
    ("lonely", "lonely solitary scene, empty space"),
    ("nostalgic", "nostalgic artwork, faded warm memories"),
    ("romantic and tender", "romantic tender artwork, soft warm light"),
    ("sensual", "sensual intimate artwork, velvet shadows"),
    ("dreamy and ethereal", "dreamy ethereal artwork, soft glow, floating"),
    ("spiritual and sacred", "spiritual sacred artwork, divine light"),
    ("peaceful and calm", "peaceful calm artwork, gentle pastel light"),
    ("hopeful", "hopeful uplifting artwork, sunrise light"),
    ("joyful and happy", "joyful happy artwork, bright vivid colors"),
    ("euphoric", "euphoric ecstatic artwork, explosive color"),
    ("playful", "playful whimsical artwork, cartoon-like"),
    ("funky and groovy", "funky groovy retro artwork, 70s colors"),
    ("energetic and danceable", "energetic dance scene, motion and lights"),
    ("futuristic and electronic", "futuristic electronic artwork, neon and chrome"),
    ("psychedelic", "psychedelic surreal artwork, swirling colors"),
    ("cinematic and dramatic", "cinematic dramatic artwork, film still"),
    ("chaotic", "chaotic frenzied artwork, fragmented"),
    ("cold and distant", "cold distant artwork, icy blue tones"),
    ("warm and cozy", "warm cozy artwork, golden lamplight"),
    ("rebellious", "rebellious punk artwork, gritty"),
    ("mournful and grieving", "mournful grieving artwork, funeral black, cold grey light"),
    ("heartbroken", "heartbroken artwork, loneliness, tears, rain"),
    ("bittersweet", "bittersweet artwork, faded warmth and sorrow"),
    ("desperate", "desperate anguished artwork, harsh shadows"),
    ("longing", "longing yearning artwork, distant figure, empty horizon"),
]
MOOD_CLAP_TEXT = [f"{m[0]} music" for m in MOODS]
MOOD_SD_TEXT = [m[1] for m in MOODS]

CLAP_TEXT = [f"music that evokes {c}" for c in CONCEPTS]
SD_TEXT = [f"{c}, artwork" for c in CONCEPTS]
