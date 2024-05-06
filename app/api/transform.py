from typing import Any
from fastapi import APIRouter

from app.models.pipeline import TransformRequest
from sklearn.pipeline import make_pipeline
import nltk

import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer

import pickle
import os

transform_router = APIRouter()

# Downloading NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords')


class StandardizeLetters:
    """
    Стандартизація літер, тобто переведення в нижній регістр
    """

    @staticmethod
    def transform(s):
        return s.lower()


class RemovePunctuation:
    """
    Корекція пунктуації
    """

    @staticmethod
    def transform(text):
        return re.sub(r'[^\w\s]', '', text)


class RemoveNumbers:
    @staticmethod
    def transform(text):
        return re.sub(r'\d', '', text)


class RemoveRareWords:
    def __init__(self, threshold=5):
        # print("Threshold is", threshold)
        self.threshold = threshold

    def transform(self, text):
        words = nltk.word_tokenize(text)
        word_freq = Counter(words)

        filtered_words = [word for word in words if word_freq[word] >= self.threshold]

        return ' '.join(filtered_words)


class Tokenize:
    @staticmethod
    def transform(text):
        return nltk.word_tokenize(text)


class FilterStopWords:
    def __init__(self, language="english"):
        self.stop_words = set(stopwords.words(language))

    def transform(self, tokens):
        return [token for token in tokens if token not in self.stop_words]


class Stemming:
    def __init__(self):
        self.stemmer = PorterStemmer()

    def transform(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]


class WhatSentiment:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()

    def transform(self, text):
        return self.sia.polarity_scores(text)


models = {}
word_features = ["the","a","and","of","to","is","in","this","br","i","with","you","that","it","was","movie","as","but","he","for","her","at","they","are","on","if","by","about","so","not","have","she","who","his","its","all","has","what","just","very","one","an","like","be","good","when","people","story","some","dont","from","do","there","little","out","film","will","time","no","show","their","can","because","which","most","also","or","him","back","woman","me","acting","first","get","does","any","along","see","had","other","after","we","character","way","where","life","probably","did","up","only","give","girl","two","hannibal","find","really","such","vampires","bad","take","buy","plot","another","think","new","lot","something","could","questions","audience","scenes","these","well","work","main","slasher","old","even","joy","today","comedy","role","didnt","films","must","part","now","thing","things","then","comes","line","love","know","cole","happen","cant","works","scifi","great","always","make","care","your","wont","thought","want","made","would","money","my","sayori","young","relationship","game","played","them","many","end","kind","totally","movies","extremely","batya","keren","mean","music","s","musicals","recent","found","appeal","leave","pathetic","should","say","learn","look","were","interesting","too","few","day","big","cheap","together","uses","instead","virus","guy","meets","doesnt","while","room","plays","leaves","someone","since","answer","never","us","around","profound","filmmakers","review","waste","screen","writer","being","actress","silly","gets","raping","watch","tyler","left","however","between","men","than","fact","angel","nothing","those","dvd","set","much","takes","ever","seen","looks","batyas","hotel","english","boat","three","maybe","ship","anyone","hear","crowd","including","why","yusuf","kerr","liked","yul","committed","stars","casey","affleck","process","expected","ultimately","genre","beyond","stupid","unrealistic","performance","grahams","roles","been","poor","shown","kept","critics","classic","enjoyed","lycans","forth","hopefully","series","sorry","how","least","problem","ideas","human","spoilers","over","enemies","during","away","leader","armor","everyone","recommend","action","weak","put","twists","based","play","real","lies","fine","job","later","important","friendship","twelve","monkeys","need","attention","traveling","named","send","cure","mental","hospital","dr","kathryn","believe","phone","call","future","seconds","keep","ask","yourself","next","may","accept","enjoy","easy","pretentious","death","laugh","critical","theyre","afraid","point","idea","poetry","genius","actually","reading","enough","mention","sandra","bullock","computer","turn","gem","disappointed","edge","kill","lines","let","choose","small","behind","friends","fumiya","surprised","resurrected","shikoku","boy","kuriyama","boring","lesbian","mind","fun","chemical","company","causing","frog","daughter","under","football","special","without","himself","ahead","poitier","presence","cassevetes","black","noordman","man","past","wants","brings","boss","characters","different","more","might","goes","side","reason","until","son","doing","casting","offer","food","outbr","themselves","hes","richards","within","scene","though","pretty","original","hole","slow","completely","ive","hand","theatre","showed","came","huge","got","dialogue","star","meaning","parents","feels","working","husband","hired","speaks","german","hebrew","toy","perfect","strange","bottle","going","saw","years","sound","especially","beatles","note","clarice","manhunter","talking","mahmut","among","cousins","wider","place","either","rogers","decade","teacher","production","isnt","brynner","heather","graham","luke","wilson","joline","determined","exhusband","midlife","crisis","lacking","delivered","reasonable","lackluster","improvementbr","blessed","misleading","allbr","pros","affleckbr","cons","predictable","ultimate","gritty","heist","elements","bogie","welles","sinatra","sweating","satisfied","comparison","upsets","proverbial","apple","cart","oceans","eleven","remake","reviewed","high","regardespecially","europe","rififi","spoke","alive","remind","younger","true","noir","isbr","criterion","commended","flawless","classy","transfer","underworld","intrigued","curious","last","feel","tight","leather","clothes","seem","besieged","victims","pest","control","act","haughty","white","pasty","humans","neat","whole","nobles","pity","abysmal","slave","thingy","feasiblebr","hit","castle","walls","jumped","steps","attack","ancient","ages","join","combat","run","fledgling","lycan","minute","werewolves","den","close","raid","armouries","use","axes","swords","vampire","inept","manages","against","romance","unconvincing","sadbr","unless","online","ticket","ok","certainly","weapons","otherwise","script","badly","sorts","inane","fantastically","written","screenplay","perceiving","perspective","timely","overdone","generally","terrific","served","greater","purpose","generating","misconception","solely","external","appearance","brothersister","amanda","bynes","tries","soccer","boys","team","finding","interest","tatum","channing","misperception","disinterested","coming","realize","attitude","pay","appreciate","deals","killed","bruce","willis","arrested","jeffrey","goines","brad","pitt","psychiatrist","railly","madeleine","stowe","disappears","chained","locked","reappears","starts","believing","coles","storiesbr","constantly","makes","message","picked","appears","throughout","therefore","better","agree","moviebr","images","dark","atmosphere","thriller","certain","helps","flaws","fictional","tells","try","storytelling","same","technique","throw","opaque","unstructured","threads","perplex","deal","topics","god","appear","sky","etc","depend","intellectually","benefit","doubt","finally","acclaimbr","fauxintellectualism","boldest","pose","begin","held","wisps","directions","raised","dropped","awkwardly","pop","cultural","references","jolting","arbitrary","stabs","missbr","intellectual","base","admit","truth","news","neither","satisfying","attempts","answers","existence","demand","wit","intellect","sadly","demonstrates","none","traits","youre","credit","forgotten","cared","read","pleasurebr","im","assuming","already","typing","amazing","brought","sympathy","programmer","often","difficult","programmerbr","anyway","basic","easily","build","subplots","upon","mainframe","into","enjoyable","premise","scarily","realistic","right","precauctions","arent","takenbr","long","shortoops","late","bringing","cutting","title","lame","nice","incorporates","taboo","viewer","move","chairs","unnecessary","choppy","novice","conjured","props","network","write","text","critique","worth","push","avoid","nonsense","counting","night","pick","else","soul","damage","subjecting","filth","hinako","moved","village","tokyo","leaving","best","returns","died","teenager","reunites","horrified","mysteriously","via","island","oh","rented","asian","horror","chiaki","nifty","unfortunately","describe","word","fruity","poorly","filmed","unimaginative","unscary","minimal","given","talk","wild","barely","bmovie","haycorny","sex","director","cody","jarrett","cheesy","slice","dumping","mutants","fish","farm","hot","kristi","russell","barbara","michaels","epa","agent","sent","investigate","environmental","dilemma","happens","enter","bartender","trixieariadne","shafferand","mansize","incites","chaos","car","crash","bosss","bleachers","stiffarm","tackling","runner","nunall","before","shot","twice","chest","antidote","ready","effectswell","rubber","costume","genitalia","prove","tough","tadpoles","still","go","bribe","witness","secrecy","highlighting","sidney","brooding","onscreen","john","city","highlywatchable","fifties","directed","martin","ritt","stunning","onlocation","photography","whitebr","delight","career","tommy","happy","outgoing","befriends","axel","suffering","selfesteem","wraps","friend","shell","introduces","family","girlfriendbr","employed","york","docks","workers","tow","knew","charles","malik","jack","warden","hardbitten","becomes","focus","discontent","leads","climactic","showdown","noordmanbr","mixed","emotions","hostile","world","name","thats","mins","funny","particular","understand","whats","happening","gonebr","yet","producers","roberto","salcedo","calls","actor","called","overactingperiod","kid","sense","favor","clue","doingbr","comedians","sometimes","tasteless","humor","itbr","dvds","taste","public","maka","pics","sell","date","funbr","valentine","cards","witty","peach","watching","buffy","reruns","samebr","cast","sizzling","display","talent","depth","denise","extras","seemed","girls","bonded","feeling","empathised","nobr","direction","managing","actual","gore","relying","imaginations","implied","threat","said","similar","manner","miss","heigel","remove","clothesbr","essentially","directorial","plus","borrowing","various","previous","flicks","psychos","shower","tributed","halloweens","masking","murdering","hiding","bodybag","far","knowbr","light","viewing","scary","jump","moments","choice","groundbreaking","playmania","basis","mel","shandi","year","pace","lasts","hours","games","period","hosts","eye","candy","hotness","destroyed","friggin","annoyingbr","players","million","times","top","surveys","minutes","worst","shows","reasons","callers","wouldnt","canceled","caught","accident","revival","packed","full","warning","bunch","short","spoofs","mood","somewhat","amusing","hysterics","biggest","princess","laia","having","cinnamon","buns","hair","head","camera","gives","grim","smile","nods","funnier","ta","chewabacca","muppet","stupidbut","couldnt","stop","laughing","drowned","laughter","wars","funnierthey","deliberately","poke","definite","went","id","telaviv","views","israels","largest","metropolis","pretentiousone","guessing","shrug","shouldersbr","protagonist","twenties","waitress","catered","weddings","evidently","walks","sea","inflatable","ring","compelled","speak","social","services","weekend","agency","closed","apartment","leaky","roof","evening","unhappy","shortcomings","performancebr","getting","married","wedding","party","course","breaks","leg","climbing","ladies","cubicle","whose","door","open","caribbean","vacation","theyve","planned","dingy","seafront","view","smells","noise","traffic","complaining","strangely","attractive","older","staying","worries","slept","strangerbr","third","filipino","crabby","mostly","concerned","philippines","asked","finds","store","plans","appearing","sort","postmodern","physical","theater","adaptation","hamlet","motherbr","storieswhich","intersect","momentarilyresolve","presumably","supposed","fantasy","element","nonexistent","somehow","inverted","sees","shop","window","effect","used","sails","billow","blown","wind","scale","reallife","draws","outline","brochure","cover","narration","womans","mentions","realized","wasnt","lose","sleep","clear","fill","finest","concerts","grew","lovin","brand","again","artist","measure","sir","p","mcs","power","spellbind","age","doco","road","band","down","roadies","live","aussie","assure","here","less","gave","almost","ago","surround","system","justice","fan","closest","concert","singer","songwriter","leadrhythm","bass","guitar","piano","ukulele","pure","entertainers","stand","alone","instrument","hold","studio","recorded","cd","raw","intended","spontaneous","excitement","emotion","crowdthis","lecter","anthony","hopkins","travesty","italy","appreciating","our","rinaldo","pazzi","giancarlo","giannini","states","julianne","moore","score","former","victim","mason","verger","gary","oldmanbr","tell","deserve","spoken","silence","lambs","truly","absolutely","badbr","near","ray","liottas","cranium","opened","forced","eat","brain","sautéed","wtf","hell","everybody","annoying","nowhere","suppose","teen","chapter","terrible","gory","gores","sake","embarrassingbr","sotl","red","dragon","risingbr","careful","gathering","details","serve","establish","personalities","passivity","emin","toprak","country","cousin","described","fear","women","chances","start","conversation","loses","decades","bachelorhood","unemployment","wellbr","case","town","hard","imagine","resentment","slackers","palpable","crumbs","expensive","carpetthe","slob","group","tarkovsky","regretbut","slight","regretthat","become","commercial","gulf","mouse","trap","theme","wonderfully","vivid","compassion","confusion","coldblooded","solving","mahmutbr","reminded","driving","each","nuts","les","chabrol","rich","hitlerian","pretensions","brialy","kiss","spider","william","hurt","figure","everybodys","nuri","bilge","ceylan","dozen","directors","active","hope","come","rely","collaborators","directing","writing","shooting","youll","personally","prefer","dancing","talents","astaire","eleanor","powell","bill","robinson","ruby","keeler","james","cagney","shirley","temple","songs","slower","dance","numbers","soapy","melodramas","offbr","caseinpoint","song","okay","deborah","minus","starred","goody","twoshoes","portrays","spends","half","threatening","siam","divorcing","myself","likesanddislikes","denying","hammerstein","folks","particularly","similarities","rh","thus","julie","andrews","flick","lavish","yes","capital","l","bigproduction","rarely","generation","dubbed","singers","unlike","able","sing","marnie","nixon","rescue","natalie","wood","west","audrey","hepburn","fair","lady","king","mongkut","stereotypical","traditionalist","portray","negative","progressive","wingers","education","anna","leonowens","straight","secularprogressives","teachers","higher","trying","cancer","employs","morebarkthanbite","justifiably","magnetism","magnificent","seven","cowboy","mesmerized","audiencebr","summary","millions","ill"]


def find_features(words):
    # words = pipeline.transform(message)
    features = {}
    for word in word_features:
        features[word] = (word in words)

    return features


class ApplyModel:
    def __init__(self, model_name):
        if model_name in models:
            self.model = models[model_name]
        else:
            file_path = os.path.join(os.path.dirname(__file__), "models", model_name + ".model")
            with open(file_path, 'rb') as file:
                self.model = pickle.load(file)
            models[model_name] = self.model

    def transform(self, words):
        # print(words)
        # print(find_features(words))
        return self.model.classify_many(find_features(words))


def create_transformer(step):
    klass = globals()[step.transformer]
    p = step.params
    if p is None:
        return klass()

    return klass(**step.params)


@transform_router.post("/transform")
async def transform(req: TransformRequest) -> Any:
    """
    Transforms the input with the provided pipeline
    """

    steps = map(create_transformer, req.steps)
    pipeline = make_pipeline(*steps)
    return pipeline.transform(req.input)
