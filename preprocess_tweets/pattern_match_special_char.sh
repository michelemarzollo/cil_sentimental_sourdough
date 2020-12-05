#!/bin/bash

for filename in $1/train*.txt; do

    # Remove repetition of special characters (except punctuation)
    sed -i "s/\([^!.?\^a-z0-9 ]\)\( \?\1\)\+/\1/gI" $filename

    # Handle smiles -> group them into <happy> / <sad> / <surprised>
    # Eyes: ; : or =
    # Tears: ,  or '
    # Nose: -
    # Mouths: ( ) [ ] / \ { } * P D O
    # Landscape smiles: ^.^ / ^_^ / ^-^ / *.* / *_* / *-* / -.- / -_- / o.o / o_o / o-o
    sed -i "s/\(^\| \)< \?3\($\| \)/\1<heart>\2/gI;
            s/\(^\| \)< \?[\/\\]\+ \?3\($\| \)/\1<broken-heart>\2/gI;
            s/\(^\| \)[>]\?[8x;:=][ ,'^oc-]*[sl\|(\\\/{[]\($\| \)/\1<sad>\2/gI;
            s/\(^\| \)\([)}sl]\|\]\)[ ,'^oc-]*[8x;:=][<]\?\($\| \)/\1<sad>\3/gI;
            s/\(^\| \)[8x;:=][ ,'^oc-]*\([)}*pd3b ]\|\]\)\($\| \)/\1<happy>\3/gI;
            s/\(^\| \)[(\/{[*pdb ]\+[ ,'^oc-]*[8x;:=]\($\| \)/\1<happy>\2/gI;
            s/\(^\| \)[\*^][ _.o-]*[\*^]\($\| \)/\1<happy>\2/gI;
            s/\(^\| \)[>-] \?[._]\+ \?[<-]\($\| \)/\1<sad>\2/gI;
            s/\(^\| \)[>.] \?[_-]\+ \?[.<]\($\| \)/\1<sad>\2/gI;
            s/\(^\| \)o[_. -]\+o\($\| \)/\1<surprised>\2/gI;
            s/\(^\| \)[8x;:=][ ,'^oc-]*o\($\| \)/\1<surprised>\2/gI;
            s/\(^\| \)o[ ,'^oc-]*[8x;:=]\($\| \)/\1<surprised>\2/gI;" $filename


    sed -i -E "s/(^|[ x.'\\\/,:+-])[0-9]+ ?( |$|[x.'\\\/,:+-]|st|nd|rd|th)/\1<number>\2/gI" $filename

    sed -i -E "s/(^|[ x.'\\\/,:+-])[0-9]+ ?( |$|[x.'\\\/,:+-]|st|nd|rd|th)/\1<number>\2/gI" $filename

    sed -i -E " s/(^| )<number> ?(th|rd|nd|st)($| )/\1<ord-number>\3/gI;
                s/(^| )<number> ?: ?<number>($| )/\1<time>\2/gI;
                s/(^| )<number> ?- ?<number>($| )/\1<range>\2/gI;
                s/(^| )<number> ?x ?<number>($| )/\1<size>\2/gI;
                s/(^| )<number> ?\/ ?<number>($| )/\1<ratio>\2/gI
                s/(^| )([\\\/|',.+-]* ?<number> ?)+($| )/\1<number>\3/gI;" $filename
    

done

# We do not want to match IDs at the beginning of the line as numbers
for filename in $1/test*.txt; do
    # Remove repetition of special characters (except punctuation)
    sed -i "s/\([^!.?\^a-z0-9 ]\)\( \?\1\)\+/\1/gI" $filename
    
    # Handle smiles -> group them into <happy> / <sad> / <surprised>
    # Eyes: ; : or =
    # Tears: ,  or '
    # Nose: -
    # Mouths: ( ) [ ] / \ { } * P D O
    # Landscape smiles: ^.^ / ^_^ / ^-^ / *.* / *_* / *-* / -.- / -_- / o.o / o_o / o-o
    sed -i "s/\(,\| \)< \?3\($\| \)/\1<heart>\2/gI;
            s/\(,\| \)< \?[\/\\]\+ \?3\($\| \)/\1<broken-heart>\2/gI;
            s/\(,\| \)[>]\?[8x;:=][ ,'^oc-]*[sl\|(\\\/{[]\($\| \)/\1<sad>\2/gI;
            s/\(,\| \)\([)}sl]\|\]\)[ ,'^oc-]*[8x;:=][<]\?\($\| \)/\1<sad>\3/gI;
            s/\(,\| \)[8x;:=][ ,'^oc-]*\([)}*pd3b ]\|\]\)\($\| \)/\1<happy>\3/gI;
            s/\(,\| \)[(\/{[*pdb ]\+[ ,'^oc-]*[8x;:=]\($\| \)/\1<happy>\2/gI;
            s/\(,\| \)[\*^][ _.o-]*[\*^]\($\| \)/\1<happy>\2/gI;
            s/\(,\| \)[>-] \?[._]\+ \?[<-]\($\| \)/\1<sad>\2/gI;
            s/\(,\| \)[>.] \?[_-]\+ \?[.<]\($\| \)/\1<sad>\2/gI;
            s/\(,\| \)o[_. -]\+o\($\| \)/\1<surprised>\2/gI;
            s/\(,\| \)[8x;:=][ ,'^oc-]*o\($\| \)/\1<surprised>\2/gI;
            s/\(,\| \)o[ ,'^oc-]*[8x;:=]\($\| \)/\1<surprised>\2/gI;" $filename

    sed -i -E "s/(,|[ x.'\\\/,:+-])[0-9]+ ?( |$|[x.'\\\/,:+-]|st|nd|rd|th)/\1<number>\2/gI" $filename

    sed -i -E "s/(,|[ x.'\\\/,:+-])[0-9]+ ?( |$|[x.'\\\/,:+-]|st|nd|rd|th)/\1<number>\2/gI" $filename

    sed -i -E " s/(,| )<number> ?(th|rd|nd|st)($| )/\1<ord-number>\3/gI;
                s/(,| )<number> ?: ?<number>($| )/\1<time>\2/gI;
                s/(,| )<number> ?- ?<number>($| )/\1<range>\2/gI;
                s/(,| )<number> ?x ?<number>($| )/\1<size>\2/gI;
                s/(,| )<number> ?\/ ?<number>($| )/\1<ratio>\2/gI
                s/(,| )([\\\/|',.+-]* ?<number> ?)+($| )/\1<number>\3/gI;" $filename

done