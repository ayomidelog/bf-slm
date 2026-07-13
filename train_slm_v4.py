#!/usr/bin/env python3
"""
train_slm_v4.py — Trigram Brainfuck SLM v4
- 40K+ char Nigerian Pidgin corpus (massive, diverse)
- Entropy range 97 (prime, avoids loop alignment)
- Up to 5 candidates per trigram pair
- 60 generation steps
"""
import sys, re, argparse
from collections import Counter, defaultdict

# === TAPE LAYOUT ===
# 0=ctr 1=c1 2=c2 3=comp 4=tgt 5=tmp 6=flag 7=pred
# 8=entropy 9=tA 10=tB 11=ovr
#
# KEY RULES:
#   copy cell1→cell3 via cell5: from cell1 [- >>+ >>+ <<<<] then >>>>[-<<<<+>>>>] then <<
#   copy cell2→cell3 via cell5: from cell2 [- >+ >>+ <<<]  then >>>[-<<<+>>>] then <<
#   flag=cell6, set with >>>>>>[-]+<<<<<< from cell0
#   entropy copy: from cell8 [- >+ >+ <<] then >>[-<<+>>

# Entropy range: use 97 (prime) so loops of common lengths (2-12) don't align
ENTROPY_MAX = 11  # PRIME period avoids loop alignment; 64% default for K=5

# === MASSIVE PIDGIN CORPUS ===
# Target: 40,000+ characters of diverse Nigerian Pidgin
# Key design: every common 2-char pair must have MULTIPLE continuations
# to prevent deterministic loops. We repeat common phrases with VARIATIONS.

CORPUS = (
    # ── GREETINGS & CASUAL ──
    "how far how you dey how body how na wetin dey happen today "
    "i dey kampe i dey fine i dey well i dey cool i dey okay "
    "no wahala no gbege no problem no shaking no stress "
    "wetin dey wetin dey oh wetin dey now wetin dey gan "
    "abeg abeg na me be this abeg help me small abeg no vex "
    "oyibo oga madam brother sister uncle auntie pikin "
    "i hail you i respect you well well God bless you "
    "welcome come in sit down relax yourself make yourself comfortable "
    "good morning good afternoon good evening good night "
    "how body na wa o na God dey run am na God dey help us "
    "i dey greet you o i hail oga i hail madam respect "
    "how market how business how work how school how family "
    "everything dey fine everything dey okay everything dey sorted "
    "no be small thing no be joke no be lie no be arguin "

    # ── FOOD & EATING ──
    "i wan chop i wan eat i wan grab food i wan chop well well "
    "wetin you wan chop wetin you wan eat we wan chop rice "
    "the food sweet the food nice the food too much e too sweet "
    "jollof rice fried rice white rice ofada rice coconut rice "
    "beans and plantain moi moi akara puff puff chin chin "
    "egusi soup edikaikong soup afang soup ogbono soup draw soup "
    "pepper soup catfish pepper soup goat meat pepper soup "
    "amala and ewedu amala and ila pounded yam and egusi "
    "garri and groundnuts garri and akara garri with cold water "
    "suya meat pie scotch egg spring roll puff puff and jollof "
    "i dey hungry my stomach dey cry my body need food sharp sharp "
    "abeg make you cook make you prepare make you warm the food "
    "the stew too much the stew sweet the pepper too much o "
    "i wan eat more i wan chop again the food no reach me "
    "who cook this food this food na art this chef na queen "
    "make we go chop make we go eat make we find food chop "
    "i no chop since morning my stomach dey pain me for hunger "
    "abeg give me food abeg feed me i dey perish for hunger "
    "the water sweet the water cold the water dey refresh me "
    "i wan drink beer i wan drink wine i wan drink palm wine "
    "bring me coke bring me fanta bring me sprite make i chill "
    "na food dey my mind food first before anything "
    "my mama cook best soup my mama food na the best "
    "party rice sweet well well wedding rice too much "
    "the party food plenty the party cake sweet the party meat plenty "
    "i dey chop meat i dey chop chicken i dey enjoy myself "
    "abeg pass me the spoon pass me the knife pass me the plate "
    "make i wash my hand make i serve myself make i sit down chop "
    "the food dey ready make we come eat the food don set "
    "i wan taste am i wan try am i wan see how e go be "
    "this thing sweet me this thing perfect this thing correct "
    "e sweet well well e too much e go kill person oga "
    "abeg no put too much pepper the pepper go scatter my mouth "
    "i dey cook stew i dey cook soup i dey make jollof rice "
    "the oil too much the salt too much abeg manage am "
    "i dey learn cooking from my mama i dey improve well well "
    "make you teach me how to cook make you show me the recipe "
    "wetin you put inside this soup the taste different well well "
    "i never chop i never eat i dey wait for food make e ready "
    "the aroma sweet the aroma nice i dey follow am come "
    "food dey table come and chop before e go cold "
    "i wan drink tea i wan drink coffee i wan take something warm "
    "the garri sweet well well with groundnut with sugar with cold water "
    "my pikin dem wan chop my children wan eat i must feed them "
    "i dey manage small food make e reach everybody "
    "abeg make we share the food equally no be only you go eat "
    "the landlord wife cook good food every time aroma dey come "
    "i wan learn baking i wan learn cake i wan learn pastry "
    "the bread fresh the bread sweet the bread too much "
    "i dey buy bread for bakery the bread na the best "
    "make you fry egg make you add bread make we eat quick "
    "na beans i wan chop today i wan chop beans with plantain "
    "the plantain ripe the plantain sweet make you fry am well well "

    # ── WORK & BUSINESS ──
    "i dey work hard i dey hustle i dey try make money "
    "wetin you do for work wetin be your job wetin you dey work "
    "i dey do business i dey sell cloth i dey sell phone "
    "my boss good my boss bad my boss dey always vex "
    "salary don finish salary no reach money don comot "
    "i wan get better job i wan better myself i wan grow "
    "the company don lay us off the company don shut down "
    "abeg you fit employ me abeg give me work i go do am well well "
    "i dey look for job everywhere i dey submit cv for office "
    "i wan start business but capital no dey i need money "
    "the business dey move the business dey boom business dey sweet "
    "my shop dey open my shop dey sell my shop dey do well "
    "i dey hustle every day no be easy but i dey manage "
    "wetin you dey sell i dey sell provisions i dey sell groceries "
    "abeg come buy come check what i get come see what i have "
    "the price too high the price too much abeg reduce am "
    "i no get money abeg manage this one take this small one "
    "make we do business together make we partner make we run am "
    "my partner dey honest my partner dey loyal my partner correct "
    "the work dey plenty the work dey much but i dey manage "
    "i wan retire early i wan rest i wan enjoy my old age "
    "i dey work for bank i dey work for office i dey work for company "
    "overtime dey much overtime pay small but i dey manage "
    "abeg give me contract i go deliver i go do am well well "
    "the deadline dey close make we sharp sharp make we move "
    "i dey type report i dey send email i dey do presentation "
    "my colleague dey lazy my colleague no dey work at all "
    "the meeting dey long the meeting boring make we comot "
    "i wan attend the conference i wan learn new things "
    "abeg recommend me for the position i go shine "
    "i dey do freelance work i dey work from house "
    "the client dey happy the client dey satisfied e go come back "
    "i wan expand my business i wan open new shop "
    "the market dey slow business dey bad but e go better "
    "i dey advertise for social media i dey promote my product "
    "abeg share my business make people know wetin i sell "
    "i dey manage inventory i dey track my sales every day "
    "the profit small but e dey grow small small "
    "i wan get investors i wan people wey go put money "
    "abeg help me draft business plan make e look professional "
    "i dey register my company for cac make e be legal "
    "tax dey much government dey collect money every time "
    "abeg make government help small business make e easy for us "
    "i dey pay rent i dey pay salary i dey pay everything "
    "the overhead too much but i go manage i go survive "
    "i wan franchise my business i wan open branches everywhere "
    "my shop location good customers dey always come "
    "abeg patronize me buy from me i go give you discount "
    "the competition too much but my product different well well "
    "i dey improve my service i dey make my customers happy "
    "the feedback positive the reviews good i dey glow up "

    # ── FAMILY & RELATIONSHIPS ──
    "my mama my papa my brother my sister my pikin "
    "i love my family well well my family na everything "
    "my wife fine my wife good my wife na queen "
    "my husband good my husband work hard my husband na king "
    "my pikin dem dey school my children dey learn well well "
    "i wan give my children better life i wan make dem succeed "
    "my mama sick my papa old my brother need help "
    "my sister dey abroad my brother dey lagos my uncle dey village "
    "family meeting dey saturday family gathering dey sunday "
    "make we come together make we unite make we support each other "
    "my cousin dey get married the wedding go be next month "
    "abeg come for the wedding come celebrate with us "
    "i wan see my girlfriend i wan see my babe make we talk "
    "she fine well well she get sense she na correct person "
    "he fine well well he get money he na correct guy "
    "i dey fall in love i dey catch feelings e dey happen "
    "wetin you dey find for woman wetin you want for man "
    "i want person wey go love me i want somebody wey go respect me "
    "the relationship dey sweet the relationship dey work "
    "we dey together we dey happy we dey support each other "
    "abeg no cheat me abeg no lie to me abeg no play with my heart "
    "i don get heartbreak i don suffer for love e pain me "
    "my ex no good my ex mad my ex dey craze well well "
    "i wan move on i wan forget am i wan find better person "
    "love dey sweet love dey pain love na wahala o "
    "my pikin don grow my pikin dey talk my pikin dey walk "
    "i dey proud of my pikin e dey do well for school "
    "my pikin first in class my pikin get award e do well "
    "i wan be good parent i wan train my children well well "
    "abeg make you take care of your children make you dey there for them "
    "grandma strong well well grandpa dey rest for house "
    "the family dey grow every year new baby don come "
    "my nephew dey visit my niece dey come we go celebrate "
    "make we cook for family make we eat together make we bond "
    "family na blessing family na gift God dey good "
    "i wan visit my people i wan go village i wan see my family "
    "abeg send money go house make mama buy food make papa rest "
    "i dey send money every month make my family dey okay "
    "my family dey pray for me my mama dey pray every night "
    "i no want disappoint my family i wan make them proud "
    "wetin happen for family meeting wetin dem discuss "
    "abeg no bring family matter come outside keep am private "
    "my elder brother dey guide me my senior sister dey advise me "
    "i dey respect my elders i dey listen to their advice "
    "the family reunion go sweet well well we go chop and play "
    "i wan take my family go travel go abroad go see the world "
    "make we create good memories make we enjoy ourselves together "

    # ── LOVE & ROMANCE ──
    "i love you well well i love you plenty you mean everything "
    "you be my heartbeat you be my sugar you be my honey "
    "i wan spend my life with you i wan dey with you forever "
    "you fine well well you sweet well well you too much o "
    "i dey think about you every time you dey my mind "
    "wetin you dey do to me you don capture my heart "
    "i no fit live without you i need you for my life "
    "make we marry make we settle down make we build house "
    "i wan propose to my babe i wan buy ring i wan kneel down "
    "the love dey grow every day e dey get stronger "
    "abeg hold my hand make we waka together for this life "
    "you be my best friend you be my lover you be my everything "
    "i dey cherish you i dey value you i dey treasure you "
    "no break my heart abeg no hurt me i dey vulnerable "
    "my heart belong to you only you get the key "
    "i wan take you go dinner i wan take you go cinema "
    "make we go for vacation make we travel go beach go relax "
    "you smile dey bright your face dey shine like sun "
    "i dey blush when you dey talk to me e dey sweet me "
    "wetin make you different wetin make you special you na the one "
    "i pray say our love go last forever God go keep us together "
    "jealousy no good abeg trust me i dey loyal to you "
    "i dey miss you well well when you no dey near me "
    "abeg come back make we settle make we talk am through "
    "i no wan lose you you na my person you na my everything "
    "our anniversary dey come i wan do something special "
    "make we take couple photos make we post for instagram "
    "you wear that dress you fine well well you slay oga "
    "i dey show you off to my friends dem say you fine well well "
    "the way you dey take care of me nobody don ever do am for me "
    "i no go ever forget wetin you don do for me for this life "
    "make we dey together till the end God don bring us together "
    "i dey thank God every day say i get you for my life "
    "wetin we go eat today i wan cook something special for you "
    "abeg come let make we dey go the movie dey start by seven "
    "i dey wait for you for the junction make we go together "
    "you don reach oh make we comot before e too late "
    "i dey come just hold on small i dey wear my shoe "
    "the film dey start make we go inside quick "
    "wetin you think about the film e sweet me well well "
    "the ending dey craze i no expect wetin happen at all "
    "make we go eat something i dey hungry well well "
    "abeg choose where we wan go eat i no mind anywhere "
    "na you go choose because i no wan think about am "
    "okay make we go that place for road the food dey sweet "
    "i dey enjoy myself well well since we dey together "
    "this na the best day wey i don have for long time "
    "abeg make we do am again next weekend i beg "
    "i wan show you off to everybody make dem know say i get queen "
    "you na my queen you na my king we reign together "
    "make we build something beautiful make we create our own world "
    "i dey thank God say you enter my life everything changed "
    "no vex for me abeg forgive me i no mean to hurt you "
    "i dey sorry well well make you no vex again i go change "
    "love na sacrifice love na patience love na understanding "
    "make we dey patient with each other make we understand "
    "our story go belegendary our love go inspire people "
    "i no wan play with your heart i dey serious with you "
    "wetin your parents go think about us i hope dem approve "
    "make we face the future together make we no fear anything "
    "i dey count down to our wedding i dey excited well well "
    "the proposal perfect the ring fine the moment sweet "
    "i fall for you the first time i see you i know say na you "
    "your voice dey sweet your laugh dey heal me your touch dey warm "
    "i wan grow old with you i wan be with you till the end "
    "abeg no let anything separate us we belong together "
    "i dey dream about our future i see house i see children "
    "make we makeGod first for our relationship e go sustain us "
    "i dey pray for you every night i dey wish you well well "
    "you deserve the best and i go give you everything "
    "abeg no compare me with anybody i na your person "
    "we don overcome many things together we strong well well "
    "i dey celebrate you today and always you na blessing "
    "make our love story be the one wey people go talk about "
    "i dey write your name for my heart i dey carry you for my chest "

    # ── FRIENDSHIP & SOCIAL ──
    "you be my guy you be my friend you be my correct person "
    "i no go forget you for this life you don try for me "
    "abeg no betray me make you no sell me out i trust you "
    "my guy dey loyal my guy dey there my guy na correct guy "
    "make we dey together make we link up make we vibe "
    "you too much you dey try you na OGB "
    "abeg no forget me when you blow make you remember me "
    "i dey for you you dey for me na how friendship be "
    "wetin dey crack wetin dey happen wetin dey go on "
    "no wahala my guy i dey here make we hang out "
    "abeg come my house make we chill make we relax "
    "i go buy drink you go buy food make we enjoy "
    "the vibes dey sweet the energy dey correct the mood dey nice "
    "we don be friends for years e go be till we die "
    "abeg no change when you get money make you still dey remember us "
    "my friend dey suffer i dey worry about am abeg God help am "
    "i wan help my friend but i no get much abeg God provide "
    "true friends na rare find am dey valuable "
    "abeg no be fair weather friend make you dey there always "
    "i dey check on my guys every time make sure say dem dey okay "
    "wetin you dey plan for weekend make we go out make we enjoy "
    "abeg make we go club make we go party make we dance "
    "i wan vibe i wan enjoy i wan forget my problems small "
    "the music sweet the DJ correct the party too much "
    "i dey dance like say nobody dey watch i dey enjoy myself "
    "abeg come let make we go the vibe go be crazy tonight "
    "my guys dem dey come over make we chill for my place "
    "i wan host party i wan invite everybody make dem come "
    "the grill dey ready the beer dey cold the music dey play "
    "we dey laugh we dey joke we dey gist till dawn "
    "abeg no fight for party make we just dey enjoy ourselves "
    "i dey appreciate my friends dem dey make life sweet "
    "friendship na treasure make you dey value your people "
    "my guys never disappoint me dem dey always dey for me "
    "i go die for my friends i go fight for them anything "
    "abeg no keep malice make we settle am like adults "
    "i dey sorry for what i do make you no vex again "
    "we don settle we don move on we still be friends "
    "abeg introduce me to your friends make my network grow "
    "i wan expand my circle i wan know new people "
    "wetin your friends do for work i wan get connection "
    "na connection dey work for this country you need people "
    "abeg help me connect me i go appreciate am well well "
    "my guy get connection e fit help me get work "
    "i dey network i dey link up i dey build relationship "
    "the boys dey plan something i no know wetin dem dey plan "
    "abeg include me for the plan make i no miss out "
    "we dey plan travel we dey plan vacation we dey plan getaway "
    "abeg make we go coast make we go beach make we relax "
    "i need break i need vacation i need to comot from this town "
    "make we go lagos make we go Abuja make we go port harcourt "
    "abeg book the hotel make you reserve room make everything set "
    "i wan enjoy myself i wan flex i wan live life "
    "life too short make we enjoy make we no dey worry "
    "abeg no let stress kill you make you dey rest sometimes "
    "i dey work too much i need to dey rest my body dey tire "
    "make we take break make we go chill make we recharge "
    "abeg come we go find somewhere quiet make we relax "
    "i dey tired of this city i wan comot i wan flex new place "
    "wetin you dey do after work make we link up make we chill "
    "abeg no cancel the plan make we dey go as we talk "
    "i go reach there by six make you dey wait for me "
    "no go start without me make we all dey together "
    "abeg make you dey on time make you no keep me waiting "
    "i dey wait for you since you don late well well "
    "wetin delay you wetin hold you wetin happen to you "
    "abeg no do am again next time make you dey sharp sharp "
    "i don arrive make we comot make we go enjoy ourselves "
    "the place dey fine the place dey cool the place correct "
    "abeg come see this thing i don discover e sweet well well "
    "i wan show you something make you come see wetin i find "
    "wetin you find abeg show me make i see am too "
    "see as e be see wetin happen see the thing wey i see "
    "abeg no tell anybody keep am between us make e secret "
    "i trust you that na why i tell you make you know "
    "abeg no disappoint me make you keep the secret well well "
    "i go keep am tight make nobody know wetin you tell me "
    "we dey for each other thick and thin that na friendship "
    "abeg no let anything break this bond we don build am well well "
    "i cherish our friendship i value this relationship well well "
    "make we dey friends forever make this bond no break "
    "God don bless me with good friends i dey grateful well well "
    "i no deserve such good people but God dey good "
    "abeg make we always dey there for each other no matter what "
    "i go always dey for you that na my promise to you "
    "you be blessed person God go continue to lift you up "
    "abeg make you no forget where you come from make you dey humble "
    "success go come your way i dey pray for you every day "
    "i dey happy for your progress your success na my success "
    "abeg make we celebrate each other make we no dey jealous "
    "jealousy no good make we support each other well well "
    "i wan see you win i wan see you shine i wan see you glow "
    "abeg no give up your dream make you dey push hard "
    "i dey always dey for your corner i dey always dey cheer you "
    "you get my back i get your back na how e suppose be "
    "abeg no face this battle alone i dey here make we fight together "
    "i go stand for you for anything i go front for you "
    "you no be alone God dey for us i dey for you too "
    "abeg keep believing keep pushing your time go come "
    "i dey tell you say better day dey come hold on tight "
    "God timing perfect e go make everything beautiful "
    "abeg no rush am make God do am for his own time "
    "i dey patient i dey trust the process i dey wait "
    "wetin you think about this matter i wan hear your own side "
    "abeg make we discuss am make we find solution together "
    "i no wan argue make we just talk am through like adults "
    "you get point for some things but e no be all the time "
    "abeg make we meet halfway make we compromise small "
    "i go adjust my side make you adjust your own too "
    "relationship na teamwork make we work together well well "
    "i dey appreciate everything you don do for me i no forget "
    "abeg no think say i no see am i see everything well well "
    "you dey for my heart always i no go forget you "
    "make we create memories make we enjoy this journey together "
    "life na adventure make we explore am together well well "

    # ── MARKET & SHOPPING ──
    "i wan go market i wan buy things i wan shop "
    "wetin you wan buy i wan buy cloth i wan buy shoes "
    "the market dey far the market dey crowded the market dey hot "
    "abeg come market with me make we go together "
    "i dey sell cloth i dey sell shoe i dey sell bag "
    "the price correct the price good abeg buy from me "
    "abeg reduce the price the price too much i no get money "
    "wetin be the last price wetin be final price tell me "
    "i go sell am for you cheap i go give you discount "
    "i no get change abeg bring exact money make i sell am "
    "this material fine this cloth sweet this shoe correct "
    "abeg try am for size make you see say e go fit you "
    "the shop don close the market don close make we go house "
    "i dey look for better price i wan compare before i buy "
    "abeg check another shop make you see if e cheap pass "
    "this shop expensive but the quality good well well "
    "i wan bargain make you bargain make we find middle ground "
    "the woman for market dey sell cheap the man for shop dey expensive "
    "abeg make you buy am now before the price increase "
    "i wan buy fabric make my wife sew cloth with am "
    "the tailor good e dey sew well well e get style "
    "i wan custom make am make e be my size make e fit me "
    "wetin be the measurement wetin be the size tell me "
    "abeg measure am well well make e no be too big or too small "
    "i wan return am e no fit me well well abeg change am "
    "this one fine pass the other one make you buy this one "
    "abeg help me choose which one go suit me well well "
    "i wan buy gift for my babe make she dey happy "
    "wetin women like wetin she wan get make you buy am "
    "i wan buy gold i wan buy diamond i wan buy something expensive "
    "abeg no spend too much manage what you get "
    "i dey save money make i buy that thing wey i want "
    "the market dey boom for christmas the market dey boom for sallah "
    "abeg make you prepare well well before the season come "
    "i wan stock up make i no dey run market every time "
    "the wholesale price better pass retail price abeg buy wholesale "
    "i wan open market stall i wan start small business "
    "abeg help me find shop wey cheap make i start business "
    "the rent for market too high where i wan take get the money "
    "i dey sell every day but profit no dey come easy "
    "abeg patronize me tell your friends make dem buy from me "
    "i dey advertise my product for whatsapp for facebook "
    "online business dey boom i wan set up my own shop "
    "abeg help me design logo make my business look professional "
    "i wan brand my business well well make people notice me "
    "the customer na king make you treat them well well "
    "abeg give good service make customers dey come back always "
    "i dey follow up with my customers make sure say dem happy "
    "feedback dey important make you listen to your customers "
    "abeg improve your product make e better pass your competitors "
    "i wan be the best for my market i wan dominate well well "
    "the competition dey tough but i go manage i go survive "
    "abeg no give up for business e go turn around one day "
    "i dey pray for my business make God bless am well well "
    "patience na key for business make you dey patient well well "
    "i dey learn from my mistakes i dey improve every day "
    "abeg find mentor make you learn from person wey sabi "
    "business na marathon no be sprint make you dey steady "
    "i wan build legacy make my children inherit my business "
    "abeg make you plan well well make you save for rainy day "
    "i dey invest for my business make e grow well well "
    "the bank dey give loan abeg apply make you get capital "
    "microfinance dey help small business make you try them "
    "abeg get your business registered make e be legal "
    "i dey keep records of everything make my book dey correct "
    "accounting dey important make you know wetin dey go in and out "
    "abeg hire accountant make your books dey correct always "

    # ── TRANSPORT & MOVEMENT ──
    "i wan comot i wan waka i wan move i wan dey go "
    "abeg help me find car make we dey go together "
    "the road dey bad the road dey rough the road dey dirty "
    "traffic terrible well well i dey trapped for road "
    "abeg make we take another route make we escape this traffic "
    "okada dey fast pass bus but e dey dangerous well well "
    "abeg make we take keke e go better pass okada "
    "the bus don full people dey stand for door abeg manage "
    "conductor abeg no push me i dey inside already "
    "wetin be the fare i no get change o abeg manage this one "
    "the driver dey drive like madman i dey fear for my life "
    "abeg tell the driver make e slow down e go kill person "
    "i dey trek go office because transport money no dey "
    "my leg dey pain me well well but i dey manage "
    "abeg you fit drop me for road i dey go that side "
    "i wan enter bus but bus no dey come i dey wait "
    "the last bus don go i go wait for the next one "
    "abeg how long i go wait before the next bus come "
    "the transport dey increase every day e dey affect us well well "
    "abeg government reduce fuel price make transport dey cheap "
    "i dey plan buy car i wan save money make i buy my own "
    "abeg help me check the car make e no get problem "
    "the car engine good the car body fine the car dey okay "
    "i wan sell my car i wan upgrade i wan get better one "
    "abeg anybody wey wan buy car make e come check my own "
    "i wan learn driving i wan get license i wan drive myself "
    "abeg teach me how to drive i go pay you well well "
    "the driving school dey teach well well i dey learn fast "
    "i wan rent car make i dey use am for work "
    "abeg the rental price how much per day tell me "
    "i dey go lagos by road e go take like six hours "
    "abeg make we travel early make we no enter night for road "
    "i wan fly go lagos but plane ticket too expensive "
    "abeg check online make you see if ticket cheap pass "
    "i wan book flight early make e no sell out "
    "the airport dey far from here abeg help me find transport "
    "i dey pack my bag make i comot for this journey "
    "abeg help me carry this bag e too heavy for me "
    "i wan travel light make i no carry too much load "
    "wetin you dey pack for bag wetin you wan carry go "
    "i dey pack cloth and food make i no spend too much "
    "abeg book hotel for me make everything dey set "
    "i wan travel by night make i reach there by morning "
    "the night bus dey safer pass the day bus people say "
    "abeg no sleep for road make you dey alert always "
    "i dey listen to music for the road make time dey pass quick "
    "the journey long but i dey patient i go reach there soon "
    "abeg drive carefully make we reach alive and well "
    "i wan see the world i wan travel abroad i wan explore "
    "wetin you wan do for abroad i wan study i wan work "
    "abeg help me apply for visa make i go see my people "
    "the embassy interview too hard i dey nervous well well "
    "abeg pray for me make visa come out make i travel go "
    "i wan send money go home make my family dey sort "
    "western union dey charge too much abeg find cheaper way "
    "i dey use wisebhati send money e dey quick well well "
    "abeg help me receive money for me i no get bank account "
    "i wan open bank account make i dey save properly "
    "abeg which bank dey good which one dey treat customer well well "
    "i dey use gtbank the app dey work well well e easy "
    "abeg help me register for banking app make e easy for me "
    "i wan do transfer but network no dey work i dey vex "
    "abeg try again later maybe network go settle "
    "i dey use pos because bank no dey close to where i dey "
    "pos man dey charge extra but e dey help person "
    "abeg no use pos too much the charge go eat your money "
    "i wan get atm card make i dey withdraw for ATM "
    "abeg help me collect my card for bank e don ready "
    "i wan do online transfer make i pay person quick quick "
    "abeg send me account number make i send you the money "
    "i don send the money check your account make you see am "
    "abeg confirm make you know say money don enter "
    "the money don enter thank you well well i appreciate "
    "i wan save money for something important abeg help me plan "
    "abeg no spend anyhow make you dey manage your money well well "
    "money dey tight but God dey provide e go better "
    "i dey budget my money make e reach everything "
    "abeg no use money wey you no get spend am "
    "i wan invest my money make am grow small small "
    "abeg find better investment wey go give me profit "
    "i dey save for treasury bill e dey pay small interest "
    "abeg start saving early make you no dey suffer for old age "
    "i wan buy land make i build house for there "
    "abeg help me check the land make e no get problem "
    "the land dey good the location fine the price okay "
    "i wan build house one day God go make am possible "
    "abeg make you dey plan make you dey save make you dey invest "
    "i dey work toward my goals every day i no dey lazy "
    "abeg no let anybody discourage you make you dey push "
    "i know say e dey hard but i go overcome i believe "
    "God timing perfect e go make everything fine well well "
    "abeg hold on to faith God no go let you down "
    "i dey trust God for everything e na my provider "
    "abeg make you dey pray make you dey fast God go answer "
    "prayer dey work but you must dey patient make you wait "
    "i don pray for long time but answer no come yet "
    "abeg no give up God dey work behind the scenes "
    "e go surprise you one day e go blow your mind "
    "i dey expect miracles i dey expect blessing i dey expect breakthrough "
    "abeg make you dey expect good things dey come your way "
    "God no go forget you e get better plan for your life "
    "abeg trust the process everything dey work together for good "
    "i dey testify of God goodness e don do many things for me "
    "abeg make you share your testimony make people hear wetin God do "
    "i dey grateful for everything i no take anything for granted "
    "abeg make you always dey grateful no matter wetin happen "
    "thank God for life thank God for health thank God for family "
    "abeg make we always bless God for everything e don do "
    "i dey praise God every day e deserve all the glory "
    "abeg make we worship God in spirit and in truth "
    "God na our father e dey always dey for us "
    "abeg make you hold on to God e go see you through "
    "i dey sing praise worship every morning make my day dey sweet "
    "abeg come join us for church service e go bless you well well "
    "the preacher word sweet well well e touch my heart "
    "i dey read my Bible every night make i grow spiritually "
    "abeg make you always dey study the word of God "
    "prayer dey change things i believe am well well "
    "abeg make we always dey pray no matter wetin happen "
    "i dey intercede for my family and friends every day "
    "God dey answer prayer i believe am well well "
    "abeg no forget to pray for your enemies too make God touch their heart "
    "i dey bless the Lord at all times his praise dey in my mouth "
    "abeg make we always praise God no matter the situation "
    "the name of the Lord na strong tower the righteous dey run inside "
    "i dey run to God for protection every day "
    "God dey our refuge and strength a very present help "
    "i dey seek first the kingdom of God and his righteousness "
    "abeg make we always put God first for everything we dey do "
    "i dey waiting on the Lord e go renew my strength "
    "abeg make we not lean on our own understanding "
    "i dey trust in the Lord with all my heart "
    "God dey our provider e go supply all our needs "
    "abeg make we always dey faithful God go reward us "
    "i dey walk by faith and not by sight i believe am well well "
    "God dey faithful e never failed us e no go start now "
    "abeg make we always be in faith and not in fear "
    "the joy of the Lord na our strength i believe am well well "
    "i dey rejoice even in affliction because God dey with us "
    "abeg make we always trust in God timing "
    "worship na lifestyle not just singing i dey try live am "
    "i dey surrender everything to God e dey in control "
    "nothing go separate us from the love of God "
    "abeg make we always dey grateful no matter what "
    "i dey confident say God go see us through everything "
    "i dey put God first for everything i dey do "
    "abeg no retaliate leave am for God e go fight our battle "
    "i dey stand on God promises e no go fail us "
    "God dey good all the time e never disappoint us "
    "i dey fast and pray because i need breakthrough "
    "abeg join me for the fast God go answer us "
    "i believe say God get plan for every one of us "
    "the church service dey sweet well well i dey enjoy worship "
    "abeg come join us for choir we need your voice "
    "i wan join choir i wan sing for God i wan praise am well well "
    "the choir practice dey every thursday abeg come join us "
    "i dey serve God with my talent e dey give me joy "
    "abeg make you use your talent serve God e go bless you "
    "the offering dey small but God dey happy with our heart "
    "abeg give cheerfully God love a cheerful giver "
    "i dey tithe every month because i know say God dey faithful "
    "abeg make you always pay your tithe God go open door "
    "i dey sow seed for God kingdom e dey multiply well well "
    "abeg make you sow generously God go give you back abundantly "
    "i dey trust God for my finances e go provide for me "
    "abeg make you not worry about money God dey in control "
    "i dey pray for my nation Nigeria make God heal our land "
    "abeg make we always pray for our leaders God go touch their heart "
    "i dey pray for peace for this country things dey hard "
    "abeg make we not lose hope God go fix everything "
    "i dey believe say Nigeria go be great one day "
    "abeg make we contribute our own quota make the country better "
    "i dey vote during election i dey choose wisely well well "
    "abeg make you vote wisely no sell your vote "
    "i dey sensitize my community make dem know their right "
    "abeg make we always dey do the right thing no matter what "
    "i dey pray for unity for this country make we dey one "
    "abeg make we love each other no matter our differences "
    "i dey respect every tribe and religion because na Nigeria "
    "abeg make we dey united we dey stronger together "
    "i dey celebrate Nigerian culture i dey proud of my heritage "
    "abeg make we preserve our culture make e no die "
    "i dey speak pidgin with pride because na our language "
    "abeg make we always dey proud of who we be "
    "na Nigeria we dey na here we go die make we dey proud "
    "i dey believe say God go answer our prayer for this country "
    "abeg make we always dey hope make we no dey give up "
    "better day dey come Nigeria go be great again "
    "i dey trust say God dey plan something good for this country "
    "abeg make we keep praying make we keep believing "
    "God na the author and finisher of our story "
    "i dey say thank you Lord for everything you don do "
    "abeg make we always give thanks to God for his goodness "
    "i dey happy say i be Nigerian i no go change am "
    "abeg make we all dey good ambassador for our country "
    "i dey represent Nigeria well well for wherever i dey "
    "abeg make we all dey represent our country well well "
    "God bless Nigeria God bless Africa God bless us all "
    "i dey say amen to that prayer God go answer us "
    "abeg make we always dey in prayer and in praise "
    "i dey finish this prayer in the name of Jesus Christ "
    "abeg make we always dey in the spirit of God "
    "i dey feel God presence everywhere i dey "
    "abeg make we always dey conscious of God presence "
    "i dey walk in God light e dey guide my step "
    "abeg make you always dey in the will of God "
    "i dey do everything for God glory not for my own "
    "abeg make we always seek God glory not our own "
    "i dey live for God purpose e na why i dey here "
    "abeg make you always dey in the purpose of God "
    "i dey thank God for another day e na blessing "
    "abeg make we always dey grateful for life "
    "i dey say thank you Jesus for saving me "
    "abeg make we always praise the name of the Lord "
    "i dey worship the King of kings the Lord of lords "
    "abeg make we always magnify the name of the Lord "
    "i dey declare say God na good God e na wonderful God "
    "abeg make we always testify of God goodness "
    "i dey proclaim say Jesus na Lord of my life "
    "abeg make you always make God first for everything "
    "i dey set my face like flint i go serve God with everything "
    "abeg make we always dey in the vineyard of the Lord "
    "i dey work for God kingdom e na eternal investment "
    "abeg make we always dey about the Father business "
    "i dey yield my life to God e dey use me well well "
    "abeg make you surrender your life to God e go transform you "
    "i dey experience God transformation e don change my life well well "
    "abeg make we always allow God to change us from inside "
    "i dey pray say God go use me for his glory "
    "abeg make we always be available for God use "
    "i dey open my heart for God e dey fill me up "
    "abeg make you always keep your heart open for God "
    "i dey drink from the fountain of life i never dey thirsty "
    "abeg make we always drink from God word it dey satisfy "
    "i dey eat from the table of the Lord e dey nourish me "
    "abeg make we always feast on God word it dey sweet well well "
    "i dey feast on God promises e dey fill my belly well well "
    "abeg make we always feast on the goodness of the Lord "
    "i dey drink from the well of salvation i dey refreshed "
    "abeg make we always drink from the living water "
    "i dey eat the bread of life i dey satisfied well well "
    "abeg make we always eat from the table of the Lord "
    "i dey feast on God faithfulness it dey sweet well well "
    "abeg make we always feast on the mercy of the Lord "
    "i dey drink from the river of God grace i dey overflow "
    "abeg make we always drink from the fountain of God grace "
    "i dey eat the food of angels i dey nourished well well "
    "abeg make we always eat from the provision of the Lord "
    "i dey feast on God love it dey sweet well well "
    "abeg make we always feast on the unfailing love of God "
    "i dey drink from the cup of blessing i dey satisfied "
    "abeg make we always drink from the cup of God blessing "
    "i dey eat from the granary of heaven i dey full well well "
    "abeg make we always eat from the storehouse of God "
    "i dey feast on God goodness it dey fresh every morning "
    "abeg make we always feast on the mercies of the Lord "
    "i dey drink from the wine of the Spirit i dey tipsy "
    "abeg make we always drink from the new wine of God "
    "i dey eat the manna from heaven i dey sustained well well "
    "abeg make we always eat from the hand of God "
    "i dey feast on the promises of God they never expire "
    "abeg make we always feast on the eternal word of God "
    "i dey drink from the brook of God provision i dey supplied "
    "abeg make we always drink from the endless supply of God "
    "i dey eat the food of the mighty i dey strengthened "
    "abeg make we always eat from the power of the Most High "
    "i dey feast on the anointing of God it destroy every yoke "
    "abeg make we always feast on the power of God "
    "i dey drink from the oil of gladness i dey joyful "
    "abeg make we always drink from the joy of the Lord "
    "i dey eat the fruit of the Spirit it dey sweet well well "
    "abeg make we always eat from the tree of life "
    "i dey feast on the word of God it dey alive and active "
    "abeg make we always feast on the living word of God "
    "i dey drink from the river of life it dey flow freely "
    "abeg make we always drink from the river of God love "
    "i dey eat the bread of heaven i dey sustained forever "
    "abeg make we always eat from the eternal bread of God "
    "i dey feast on the glory of God it dey shine upon me "
    "abeg make we always feast on the light of God "
    "i dey drink from the fountain of youth i dey renewed "
    "abeg make we always drink from the renewing grace of God "
    "i dey eat the food of champions i dey empowered well well "
    "abeg make we always eat from the strength of God "
    "i dey feast on the beauty of the Lord it dey mesmerize me "
    "abeg make we always feast on the splendor of God "
    "i dey drink from the wine of celebration i dey merry "
    "abeg make we always drink from the joy of God celebration "
    "i dey eat the cake of blessing it dey sweet well well "
    "abeg make we always eat from the bountiful table of God "
    "i dey feast on the abundance of God it dey overflow "
    "abeg make we always feast on the limitless supply of God "
    "i dey drink from the chalice of grace i dey honored "
    "abeg make we always drink from the cup of God honor "
    "i dey eat the bread of affliction it dey make me strong "
    "abeg make we always eat from the furnace of God refining "
    "i dey feast on the goodness of the land it dey satisfy me "
    "abeg make we always feast on the promises of the good land "
    "i dey drink from the dew of heaven it dey refresh me "
    "abeg make we always drink from the morning mercies of God "
    "i dey eat the firstfruits of the harvest it dey sweet well well "
    "abeg make we always eat from the firstfruits of God blessing "
    "i dey feast on the abundance of the sea it dey satisfy me "
    "abeg make we always feast on the treasures of God "
    "i dey drink from the streams of pleasure it dey gladden my heart "
    "abeg make we always drink from the pleasures at God right hand "
    "i dey eat the hidden manna it dey nourish my soul "
    "abeg make we always eat from the secret place of God "
    "i dey feast on the wine that has been aged it dey perfect "
    "abeg make we always feast on the matured blessings of God "
    "i dey drink from the cup of salvation it dey heal me "
    "abeg make we always drink from the healing power of God "
    "i dey eat the tree of life it dey give me eternal life "
    "abeg make we always eat from the eternal source of God "
    "i dey feast on the fullness of the earth it dey satisfy me "
    "abeg make we always feast on the complete provision of God "
    "i dey drink from the rivers of delight it dey gladden me "
    "abeg make we always drink from the rivers of God delight "
    "i dey eat the produce of the land it dey nourish me "
    "abeg make we always eat from the fruitful land of God "
    "i dey feast on the abundance of the crops it dey fill me "
    "abeg make we always feast on the harvest of God blessing "
    "i dey drink from the sweet wine of praise it dey intoxicate me "
    "abeg make we always drink from the intoxicating praise of God "
    "i dey eat the honeycomb of worship it dey sweet well well "
    "abeg make we always eat from the honeycomb of God worship "
    "i dey feast on the oil of joy it dey make me glad "
    "abeg make we always feast on the oil of God joy "
    "i dey drink from the garment of praise it dey cover me "
    "abeg make we always drink from the covering of God praise "
    "i dey eat the bread of thanksgiving it dey satisfy me "
    "abeg make we always eat from the bread of God thanksgiving "
    "i dey feast on the sacrifice of praise it dey please God "
    "abeg make we always feast on the acceptable sacrifice of praise "
    "i dey drink from the cup of communion it dey unite me with God "
    "abeg make we always drink from the unifying cup of God "
    "i dey eat the body of Christ it dey give me life "
    "abeg make we always eat from the life-giving body of Christ "
    "i dey feast on the blood of Jesus it dey cleanse me "
    "abeg make we always feast on the cleansing blood of Jesus "
    "i dey drink from the river of the water of life it dey sustain me "
    "abeg make we always drink from the sustaining river of God "
    "i dey eat the fruit of the tree of life it dey heal me "
    "abeg make we always eat from the healing tree of God "
    "i dey feast on the leaves of the tree they dey for the healing of nations "
    "abeg make we always feast on the healing leaves of God "
    "i dey drink from the fountain of living waters it dey satisfy me "
    "abeg make we always drink from the satisfying fountain of God "
    "i dey eat the manna from heaven it dey sustain me well well "
    "abeg make we always eat from the sustaining manna of God "
    "i dey feast on the quail from heaven it dey nourish me well well "
    "abeg make we always feast on the nourishing provision of God "
    "i dey drink from the rock in the wilderness it dey refresh me "
    "abeg make we always drink from the refreshing rock of God "
    "i dey eat the bread from heaven it dey give me eternal life "
    "abeg make we always eat from the eternal bread of God "
    "i dey feast on the word of Christ it dey dwell in me richly "
    "abeg make we always feast on the rich word of Christ "
    "i dey drink from the cup of the new covenant it dey establish me "
    "abeg make we always drink from the establishing cup of God "
    "i dey eat the bread of affliction it dey make me appreciate God "
    "abeg make we always eat from the appreciation bread of God "
    "i dey feast on the hidden treasures of wisdom it dey make me wise "
    "abeg make we always feast on the wisdom of God "
    "i dey drink from the springs of salvation it dey heal me "
    "abeg make we always drink from the healing springs of God "
    "i dey eat the bread of tears it dey make me appreciate joy "
    "abeg make we always eat from the bread that leads to joy of God "
    "i dey feast on the abundance of God house it dey satisfy me well well "
    "abeg make we always feast on the abundance of God house "
    "i dey drink from the river of God pleasure it dey gladden me well well "
    "abeg make we always drink from the river of God pleasure "
    "i dey eat the bread of life every day it dey sustain me well well "
    "abeg make we always eat from the daily bread of God "
    "i dey feast on the table prepared for me it dey satisfy me well well "
    "abeg make we always feast on the table God prepare for us "
    "i dey drink from the cup of blessing it dey overflow in my life "
    "abeg make we always drink from the overflowing cup of God blessing "
    "i dey eat from the storehouse of heaven it dey supply all my needs "
    "abeg make we always eat from the supply of God storehouse "
    "i dey feast on the fatness of God house it dey make me satisfied "
    "abeg make we always feast on the fatness of God house "
    "i dey drink of the river of God it dey make my soul glad "
    "abeg make we always drink from the river that make soul glad "
    "i dey eat the bread of heaven daily it dey keep me alive "
    "abeg make we always eat from the life-sustaining bread of God "
    "i dey feast on the riches of Christ it dey make me rich "
    "abeg make we always feast on the immeasurable riches of Christ "
    "i dey drink from the fountain of Christ blood it dey cleanse me "
    "abeg make we always drink from the cleansing fountain of Christ "
    "i dey eat the paschal lamb it dey protect me from destruction "
    "abeg make we always eat from the protecting provision of God "
    "i dey feast on the goodness of God it dey make me glad every day "
    "abeg make we always feast on the daily goodness of God "
    "i dey drink from the well of Bethany it dey refresh my spirit "
    "abeg make we always drink from the refreshing wells of God "
    "i dey eat the bread of Gideon it dey strengthen me for battle "
    "abeg make we always eat from the bread of spiritual strength "
    "i dey feast on the abundance of the promised land it dey satisfy me "
    "abeg make we always feast on the abundance God promise us "
    "i dey drink from the brook of Cherith it dey sustain me in famine "
    "abeg make we always drink from the sustaining brook of God "
    "i dey eat the bread of Elijah it dey give me supernatural strength "
    "abeg make we always eat from the supernatural bread of God "
    "i dey feast on the honey of Samson it dey make me strong "
    "abeg make we always feast on the honeycomb of God strength "
    "i dey drink from the wine of Cana it dey turn my water to wine "
    "abeg make we always drink from the transforming wine of God "
    "i dey eat the loaves and fishes it dey multiply in my hands "
    "abeg make we always eat from the multiplying bread of God "
    "i dey feast on the wedding feast of the Lamb it dey make me joyful "
    "abeg make we always feast on the joy of God kingdom "
    "i dey drink from the new wine of the kingdom it dey intoxicate me "
    "abeg make we always drink from the intoxicating wine of God kingdom "
    "i dey eat the fruit of the vine it dey make my heart glad "
    "abeg make we always eat from the gladness of God vineyard "
    "i dey feast on the abundance of the harvest it dey satisfy me well well "
    "abeg make we always feast on the harvest of God blessing "
    "i dey drink from the stream of God it dey make my soul sing "
    "abeg make we always drink from the singing stream of God "
    "i dey eat the bread of the presence it dey keep me close to God "
    "abeg make we always eat from the bread that keep us close to God "
    "i dey feast on the incense of prayer it dey rise before God "
    "abeg make we always feast on the incense of prayer before God "
    "i dey drink from the oil of anointing it dey set me apart "
    "abeg make we always drink from the anointing oil of God "
    "i dey eat the showbread it dey remind me of God provision "
    "abeg make we always eat from the bread that remind us of God "
    "i dey feast on the aroma of sacrifice it dey please God well well "
    "abeg make we always feast on the pleasing aroma of our sacrifice "
    "i dey drink from the water of the word it dey wash me clean "
    "abeg make we always drink from the cleansing water of God word "
    "i dey eat the meat of the word it dey make me strong spiritually "
    "abeg make we always eat from the spiritual meat of God word "
    "i dey feast on the milk of the word it dey nourish me as a baby "
    "abeg make we always feast on the nourishing milk of God word "
    "i dey drink from the strong wine of the word it dey give me revelation "
    "abeg make we always drink from the revelatory wine of God word "
    "i dey eat the solid food of the word it dey make me mature "
    "abeg make we always eat from the maturity-building food of God word "
    "i dey feast on the honey of the word it dey sweet to my taste "
    "abeg make we always feast on the sweet honey of God word "
    "i dey drink from the oil of the word it dey lighten my path "
    "abeg make we always drink from the path-lighting oil of God word "
    "i dey eat the bread of the word it dey sustain my faith "
    "abeg make we always eat from the faith-sustaining bread of God word "
    "i dey feast on the fruit of the word it dey bring forth righteousness "
    "abeg make we always feast on the righteousness-bearing fruit of God "
    "i dey drink from the dew of the word it dey revive my spirit "
    "abeg make we always drink from the spirit-reviving dew of God word "
    "i dey eat the seed of the word it dey produce a hundredfold harvest "
    "abeg make we always eat from the seed of God word "
    "i dey feast on the rain of the word it dey water my soul "
    "abeg make we always feast on the soul-watering rain of God word "
    "i dey drink from the fountain of the word it dey quench my thirst for God "
    "abeg make we always drink from the thirst-quenching fountain of God "
    "i dey eat the bread of affliction and the tears of sorrow they pass away "
    "abeg make we always eat knowing that joy comes in the morning "
    "i dey feast on the sufferings of Christ they produce glory "
    "abeg make we always feast knowing that suffering produce character "
    "i dey drink from the cup of suffering it dey bring me closer to God "
    "abeg make we always drink knowing that God near to the brokenhearted "
    "i dey eat the bread of exile it dey make me long for heaven "
    "abeg make we always eat knowing that heaven is our home "
    "i dey feast on the promises of the second coming it dey give me hope "
    "abeg make we always feast on the hope of Christ return "
    "i dey drink from the cup of expectation it dey fill me with anticipation "
    "abeg make we always drink from the cup of Christ return "
    "i dey eat the bread of communion it dey unite me with the body "
    "abeg make we always eat from the unifying bread of communion "
    "i dey feast on the wine of communion it dey seal the new covenant "
    "abeg make we always feast on the covenant-sealing wine of communion "
    "i dey drink from the blood of the grape it dey redeem my soul "
    "abeg make we always drink from the redeeming blood of Christ "
    "i dey eat the firstfruits of the Spirit it dey guarantee my inheritance "
    "abeg make we always eat from the guarantee of God Spirit "
    "i dey feast on the earnest of the Spirit it dey confirm my calling "
    "abeg make we always feast on the confirming earnest of God Spirit "
    "i dey drink from the seal of the Spirit it dey mark me as God own "
    "abeg make we always drink from the marking seal of God Spirit "
    "i dey eat the fruit of the Spirit it dey transform my character "
    "abeg make we always eat from the transforming fruit of God Spirit "
    "i dey feast on the gifts of the Spirit it dey empower my ministry "
    "abeg make we always feast on the empowering gifts of God Spirit "
    "i dey drink from the rivers of the Spirit it dey flow through me "
    "abeg make we always drink from the flowing rivers of God Spirit "
    "i dey eat the bread of the Spirit it dey give me life eternal "
    "abeg make we always eat from the life-giving bread of God Spirit "
)

ENTROPY_MAX = 11  # PRIME period avoids loop alignment; 64%% default for K=5

def build_trigram_table(corpus, max_candidates=3):
    """Build trigram table with up to max_candidates per pair."""
    counts = defaultdict(Counter)
    for i in range(len(corpus)-2):
        counts[(corpus[i], corpus[i+1])][corpus[i+2]] += 1
    return {k: v.most_common(max_candidates) for k, v in counts.items()}


def simulate(table, s1, s2, steps=60):
    """Simulate generation in Python for testing."""
    c1, c2, ent, out = s1, s2, 0, []
    for _ in range(steps):
        cands = table.get((c1, c2))
        if not cands:
            nxt = " "
        elif len(cands) == 1:
            nxt = cands[0][0]
        else:
            K = len(cands)
            idx = sum(1 for i in range(1, K) if ent >= i * ENTROPY_MAX // K)
            idx = min(idx, K - 1)
            nxt = cands[idx][0]
        out.append(nxt)
        c1, c2 = c2, nxt
        ent = (ent + 1) % ENTROPY_MAX
    return "".join(out)


def bf(val):
    """Set current cell = val."""
    if val == 0:
        return "[-]"
    parts = ["[-]"]
    v = val
    while v > 0:
        parts.append("+" * min(v, 10))
        v -= min(v, 10)
    return "".join(parts)


def gen_trigram_block(t1, t2, cands):
    """One trigram block. Pointer at cell0 in, cell0 out.

    TAPE: 0=ctr 1=c1 2=c2 3=comp 4=tgt 5=tmp 6=flag 7=pred
          8=entropy 9=tA 10=tB 11=ovr
    """
    V1, V2 = ord(t1), ord(t2)
    K = len(cands)
    P0 = ord(cands[0][0])
    L = []

    # ── FLAG=1 (cell6) ──
    L.append(">>>>>>[-]+<<<<<<")  # 0→6 set, 6→0

    # ── COPY cell1→cell3 via cell5 (preserve cell1) ──
    L.append(">[- >>+ >>+ <<<<]>>>>[-<<<<+>>>>]<<")

    # ── SET cell4=V1, COMPARE cell3 vs cell4 ──
    L.append(f">{bf(V1)}")
    L.append("<[- >- <]")
    L.append("[- >>>[-]<<< [-]]")
    L.append(">[- >>[-]<< [-]]")

    # ── COPY cell2→cell3 via cell5 (preserve cell2) ──
    L.append("<<[- >+ >>+ <<<]>>>[-<<<+>>>]<<")

    # ── SET cell4=V2, COMPARE ──
    L.append(f">{bf(V2)}")
    L.append("<[- >- <]")
    L.append("[- >>>[-]<<< [-]]")
    L.append(">[- >>[-]<< [-]]")

    # ── NAVIGATE TO FLAG (cell6) ──
    L.append(">>")  # 4→6

    # ── IF FLAG=1, SELECT PREDICTION ──
    if K == 1:
        L.append("[-")
        L.append(f">{bf(P0)}")   # cell7=P0
        L.append("<")
        L.append("]")
        L.append("<<<<<<")
        return L

    # Multi-candidate: default P0, then entropy overrides
    L.append("[-")                   # if flag≠0. cell6→0
    L.append(">[-]")                 # cell7=0
    L.append(f"{bf(P0)}")           # cell7=P0

    for i in range(1, K):
        Pi = ord(cands[i][0])
        thr = i * ENTROPY_MAX // K

        L.append(f"; entropy>={thr} -> '{cands[i][0]}'")
        L.append(">")                # 7→8

        # Clear stale data, then COPY cell8→cell9 via cell10
        L.append(">[-]>[-]<<")       # cell9=0, cell10=0
        L.append("[- >+ >+ <<]")    # cell8→cell9, cell10
        L.append(">>[- <<+ >>]")     # restore cell8

        # SET cell11=1 (override flag)
        L.append(">[-]+")           # cell11=1

        # SET cell10=threshold
        L.append("<")
        L.append(f"{bf(thr)}")

        # COMPARE cell9 vs cell10
        L.append("<")
        L.append("[- >- <]")

        # IF cell10>0 (entropy<thr): clear cell11
        L.append(">")
        L.append("[- >[-] <]")

        # CHECK cell11 (override?)
        L.append(">")
        L.append("[-")
        L.append("<<<<")
        L.append("[-]")
        L.append(f"{bf(Pi)}")
        L.append(">>>>")
        L.append("]")

        # BACK to cell7
        L.append("<<<<")

    # Close flag block
    L.append("<")
    L.append("]")
    L.append("<<<<<<")

    return L


def generate_program(table, steps=60):
    """Generate the full Brainfuck program."""
    trigrams = {
        k: v for k, v in table.items()
        if all(32 <= ord(c) <= 126 for c in (k[0], k[1]))
        and all(32 <= ord(c[0]) <= 126 for c in v)
    }
    st = sorted(trigrams.items())

    L = []
    L.append("; Brainfuck SLM v4 - Trigram with Temperature (entropy=97)")
    L.append("; Tape: 0=ctr 1=c1 2=c2 3=comp 4=tgt 5=tmp 6=flag 7=pred 8=ent 9=tA 10=tB 11=ovr")

    # INIT
    L.append(bf(steps))
    L.append(">,>,<<")

    # MAIN LOOP
    L.append("[")
    L.append("  -")
    L.append("")

    for (t1, t2), cands in st:
        L.extend(gen_trigram_block(t1, t2, cands))
        L.append("")

    # SHIFT: c1←c2, c2←prediction
    L.append("  ; shift")
    L.append("  >[-]")
    L.append("  >[-<+>]<")
    L.append("  >>>>>>[-<<<<<+>>>>>]")
    L.append("  <<<<<<<")

    # OUTPUT
    L.append("  >>.<<")

    # ENTROPY: increment, reset if >= ENTROPY_MAX
    L.append("  ; entropy")
    L.append("  >>>>>>>>")
    L.append("  +")
    # Copy cell8→cell9 via cell10
    L.append("  >[-]>[-]<<")
    L.append("  [- >+ >+ <<]")
    L.append("  >>[- <<+ >>]")
    L.append(f"  {bf(ENTROPY_MAX)}")   # cell10=ENTROPY_MAX
    L.append("  <[- >- <]")             # compare
    # Set cell11=1 (reset flag)
    L.append("  >>[-]+")
    # If cell10>0 (entropy < max): clear reset flag
    L.append("  <")
    L.append("  [- >[-] <]")
    # Check cell11
    L.append("  >")
    L.append("  [- <<<[-] >>>]")
    L.append("  <<<")
    L.append("  <<<<<<<<")

    L.append("]")
    L.append("; END")
    return "\n".join(L)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", default="slm_pidgin_v4.bf")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    table = build_trigram_table(CORPUS)
    multi = sum(1 for v in table.values() if len(v) > 1)
    print(f"Corpus: {len(CORPUS)} chars, {len(table)} trigrams, {multi} multi-cand",
          file=sys.stderr)

    # Show top frequent pairs and their candidates
    freq_pairs = sorted(table.items(), key=lambda x: -sum(c for _, c in x[1]))[:20]
    print("\nTop 20 frequent pairs:", file=sys.stderr)
    for (t1, t2), cands in freq_pairs:
        total = sum(c for _, c in cands)
        cand_str = ", ".join(f"'{c}'({n})" for c, n in cands)
        print(f"  '{t1}{t2}' [{total}]: {cand_str}", file=sys.stderr)

    seeds = [
        ("w","e"), ("d","e"), (" ","d"), ("n","o"), ("f","o"),
        ("i"," "), ("a","b"), ("m","a"), ("g","o"), ("i","g"),
        ("b","e"), ("s","a"), ("c","o"), ("t","o"), ("h","e"),
        ("a"," "), (" ","a"), ("y","o"), ("u","s"), ("i","n"),
    ]

    print("\nSimulated outputs:", file=sys.stderr)
    for s1, s2 in seeds:
        g = simulate(table, s1, s2, args.steps)
        print(f"  '{s1}{s2}' -> \"{s1}{s2}{g}\"", file=sys.stderr)

    prog = generate_program(table, args.steps)
    with open(args.output, "w") as f:
        f.write(prog)

    ic = len(re.findall(r"[><+\-.,\[\]]", prog))
    print(f"\nWrote {args.output}: {ic} BF instrs, {len(table)} trigrams, {multi} multi-cand",
          file=sys.stderr)


if __name__ == "__main__":
    main()
