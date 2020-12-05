"""
    Helper file that allows to more easily create and edit the sed script used to perform pattern matching.
    Note that pattern matching operates in place in the directory tweet_tmp1_dir (see preproc_config.py)
"""
from preproc_config import *

remove_too_many_duplicates = ("([a-z])\\1{2,}", "\\1\\1")
                # Contracted forms of verbs and negations
change_ops = [  ("(r)","are"),
                ("(m)","am"),
                ("(i ?'? ?m)","i am"),
                ("(i ?'ll)","i will"),
                ("(i ?'? ?d)","i would"),
                ("(it ?'? ?s)","it is"),
                ("(how ?'? ?s)","how is"),
                ("(who ?'? ?s)","who is"),
                ("(she ?'? ?s)","she is"),
                ("(he ?'? ?s)","he is"),
                ("(here ?'? ?s)","here is"),
                ("(there ?'? ?s)","there is"),
                ("(where ?'? ?s)","where is"),
                ("(that ?'? ?s)","that is"),
                ("(why ?'? ?s)","why is"),
                ("(what ?'? ?s)","what is"),
                ("(when ?'? ?s)","when is"),
                ("(an ?' ?t)","am not"),
                ("(isn ?' ?t|isnt|ain ?' ?t|aint|ain ?' ?tt)","is not"),
                ("(aren ?' ?t|arent|arn ?' ?t)","are not"),
                ("(wasn ?' ?t|wasnt|wan ?' ?t|wasen ?' ?t|wasent|wsn ?' ?t|wasn ?' ?tt|wasn ?' ?ttt)","was not"),
                ("(weren ?' ?t|werent|wern ?' ?t|wernt|wren ?' ?t)","were not"),
                ("(haven ?' ?t|havent|havn ?' ?t|havnt|haen ?' ?t|hven ?' ?t|haven ?' ?tt)","have not"),
                ("(i ?' ?vn ?' ?t)","i have not"),
                ("(hasn ?' ?t|hasnt)","has not"),
                ("(hadn ?' ?t|hadnt|haden ?' ?t|hadent)","had not"),
                ("(don ?' ?t|dont|dn ?' ?t|donn ?' ?t|don ?' ?tt|don ?' ?ttt)","do not"),
                ("(idon ?' ?t|idont)","i do not"),
                ("(doesn ?' ?t|doesnt|dosen ?' ?t|dosent|dosn ?' ?t|dosnt|doen ?' ?t|doesen ?' ?t|doesn ?' ?tt)","does not"),
                ("(didn ?' ?t|didnt|din ?' ?t|dint|diden ?' ?t|dident|diddn ?' ?t|diddnt|ddidn ?' ?t|didin ?' ?t|didn ?' ?tt)","did not"),
                ("(ididn ?' ?t|ididnt|ididn ?' ?tt)","i did not"),
                ("(can ?' ?t|cant|cannot|cn ?' ?t|cann ?' ?t|can ?' ?tt|can ?' ?ttt)","can not"),
                ("(ican ?' ?t|icant|icannot|icn ?' ?t)","i can not"),
                ("(won ?' ?t|wont|won ?' ?tt)","will not"),
                ("(shan ?' ?t)","shall not"),
                ("(wouldn ?' ?t|wouldnt|wudn ?' ?t|wudnt|woudn ?' ?t|woudnt|wldn ?' ?t|wudln ?' ?t|woulden ?' ?t|woudln ?' ?t)","would not"),
                ("(iwouldn ?' ?t)","i would not"),
                ("(couldn ?' ?t|couldnt|cldn ?' ?t|couldn ?' ?tt)","could not"),
                ("(shouldn ?' ?t|shouldnt|shuldn ?' ?t|shuldnt|shoudn ?' ?t|shudn ?' ?t)","should not"),
                ("(mustn ?' ?t|mustnt)","must not"),
                ("(mightn ?' ?t)","might not"),
                ("(needn ?' ?t|neednt)","need not"),
                ("(theren ?' ?t|therent)","there is not"),
                
                # Slang and abbreviations
                ("(gonna|gona)", "going to"),
                ("(wanna|wana)", "want to"),
                ("(on it)","on it"),
                ("(afaik)", "as far as i know"),
                ("(asap)", "as soon as possible"),
                ("(wtf)", "what the fuck"),
                ("(irl)", "in real life"),
                ("(imo)", "in my opinion"),
                ("(imho)", "in my humble opinion"),
                ("(u|ya)", "you"),
                ("(ur|u're)", "you are"),
                ("(u2)", "you too"),
                ("(l8r)", "later"),
                ("(m8)", "mate"),
                ("(sk8)", "skate"),
                ("(w8)", "wait"),
                ("(gr8)", "great"),
                ("(g9)", "genius"),
                ("(oic)", "oh i see"),
                ("(pita)", "pain in the ass"),
                ("(prt)", "party"),
                ("(4get)", "forget"),
                ("(4gets)", "forgets"),
                ("(b4)", "before"),
                ("(b4n)", "bye for now"),
                ("(pls|plz)", "please"),
                ("(sm)", "some"),
                ("(nw)", "now"),
                ("(usd)", "used"),
                ("(thn)", "then"),
                ("(anthr)", "another"),
                ("(z)", "is"),
                ("(reli)", "really"),
                ("(excitd)", "excited"),
                ("(ppl)", "people"),
                ("(thx|ty|thank-you)", "thank you"),
                ("(cu|cya)", "see you"),
                ("(ic)", "i see"),
                ("(cul8r)", "see you later"),
                ("(cuz)", "because"),
                ("(reasn)", "reason"),
                ("(abt)", "about"),
                ("(dunno)", "do not know"),
                ("(followme)", "follow me"),
                ("(followback)", "follow back"),
                ("(wdym)", "what do you mean"),
                ("(gg)", "good game"),
                ("(gn)", "good night"),
                ("(wb)", "welcome back"),
                ("(7k)", "sick:-d laugher"),
                ("(atk)", "at the keyboard"),
                ("(atm)", "at the moment"),
                ("(a3)", "anytime, anywhere, anyplace"),
                ("(bak)", "back at keyboard"),
                ("(bbl)", "be back later"),
                ("(bbs)", "be back soon"),
                ("(bfn|b4n)", "bye for now"),
                ("(brb)", "be right back"),
                ("(brt)", "be right there"),
                ("(btw)", "by the way"),
                ("(fc)", "fingers crossed"),
                ("(fwiw)", "for what it's worth"),
                ("(fyi)", "for your information"),
                ("(gal)", "get a life"),
                ("(gmta)", "great minds think alike"),
                ("(ilu|ily)", "i love you"),
                ("(iow)", "in other words"),
                ("(ldr)", "long distance relationship"),
                ("(ltns)", "long time no see"),
                ("(mte)", "my thoughts exactly"),
                ("(nrn)", "no reply necessary"),
                ("(prw)", "parents are watching"),
                ("(qpsa ?)", "que pasa ?"),
                ("(ttfn)", "ta-ta for now !"),
                ("(ttyl)", "talk to you later"),
                ("(u4e)", "yours for ever"),
                ("(wtg)", "way to go !"),
                ("(wuf)", "where are you from ?"),
                ("(brotha)", "brother"),
                ("(bday)", "birthday"),
                ("(yes's)", "yeses"),
                ("(no's)", "noes"),
                ("(gj)", "good job"),
                ("(np)", "no problem"),
                ("(iloveyou)", "i love you"),
                ("(actualy)", "actually"),
                ("(againn)", "again"),
                ("(agian)", "again"),
                ("(abou)", "about"),
                ("(culd)", "could"),
                ("(lookin)", "looking"),
                ("(comin)", "coming"),
                ("(workin)", "working"),
                ("(looved)", "loved"),
                ("(thatt)", "that"),
                ("(backcan't)", "back can not"),
                ("(idk)", "i do not know"),
                ("(idc)", "i do not care"),
                ("(nsfw)", "not safe for work"),
                ("(till)", "until"),
                ("(til)", "until"),
                ("(imma)", "i am going to"),
                ("(sry|srry|sory)", "sorry"),
                ("(kinda)", "kind of"),

                # Punctuation
                ("(\\! ?){2,}", "\\!\\!\\!"),
                ("(\\? ?){2,}", "\\?\\?\\?"),   
                ("(\\. ?){2,}", "\\.\\.\\."),
                ("(4[\\.:,\\/]20)","<weed>"),

                # Laughters
                ("(x ?){2,}", "xxx"),
                ("(o*x+o+[xo]*)","<xoxo>"),
                ("(a*h+a+[ah]+)","<ahah>"),
                ("(a*j+a+[aj]+)","<ajaj>"),
                ("(e*h+e+[eh]+)","<eheh>"),
                ("(u*h+u+[uh]+)","<uhuh>"),
                ("(y*a+[ay]+)","<yaya>"),
                ("(a+w+[awh]*)","<awwh>"),
                ("(m+u?[ah]+w*h+[awh]*)","<muah>"),
                ("([bp]+u?[ah]+[ah]+[ah]*)","<buah>"),
            ]
            
change_suffix_ops = [   ("(n't|n'tt|n'ttt)", "not"),
                        ("('s)", "<genitive>"),
                        ("('d)", "would"),
                        ("('ll)", "will"),
                        ("('re)", "are"),
                        ("('ve)", "have"),
            ]

with open('pattern_matching.sh', 'w') as f:
    f.write("""\
#!/bin/bash

pids=\"\"

for filename in $1/train*.txt; do
""")
    ########################
    # Process training set #
    ########################

    # Transform triplets of letters into duplicates detected by next pattern matching
    op_str = "s/{0:}/{1:}/gI;".format(remove_too_many_duplicates[0], remove_too_many_duplicates[1])

    
    for find,replace in change_ops:
        op_str += "s/(^| ){0:}($| )/\\1{1:}\\3/gI;".format(find, replace)
#        f.write("\tsed -i \"s/ {0:} / {1:} /gI; s/^{0:} /{1:} /gI; s/ {0:}$/ {1:}/gI\" $filename\n".format(find, replace))

    for find,replace in change_suffix_ops:
        op_str += "s/{0:}($| )/ {1:}\\2/gI;".format(find, replace)

    f.write("\tsed -i -E \"{}\" $filename &\n".format(op_str))
    f.write("\tpids=\"$pids $!\"\n")
    f.write("done\n")
    
    ########################
    #   Process test set   #
    ########################
    op_str = ""

    # Transform triplets of letters into duplicates detected by next pattern matching
    # NOTE: no need to worry about IDs because we are only matching letters
    op_str = "s/{0:}/{1:}/gI;".format(remove_too_many_duplicates[0], remove_too_many_duplicates[1])

    #op_str = ""

    for find,replace in change_ops:
        op_str += "s/(^|,| ){0:}($| )/\\1{1:}\\3/gI;".format(find, replace)
    for find,replace in change_suffix_ops:
        op_str += "s/{0:}($| )/ {1:}\\2/gI;".format(find, replace)
    
    f.write("\n")
    f.write("sed -i -E \"{}\" $1/test_data.txt &\n".format(op_str))
    f.write("\tpids=\"$pids $!\"\n")
    
    f.write("wait $pids\n")
