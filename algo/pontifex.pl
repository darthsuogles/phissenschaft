#!/usr/bin/perl -s

## Solitaire cryptosystem: verbose version
## Ian Goldberg <ian@cypherpunks.ca>, 19980817

## Make sure we have at least the key phrase argument
die "Usage: $0 [-d] 'key phrase' [infile ...]\n" unless $#ARGV >= 0;

## Set the multiplication factor to -1 if "-d" was specified as an option
## (decrypt), or to 1 if not.  This factor will be multiplied by the output
## of the keystream generator and added to the input (this has the effect
## of doing addition for encryption, and subtraction for decryption).
$f = $d ? -1 : 1;

## Set up the deck in sorted order.  chr(33) == '!' represents A of clubs,
## chr(34) == '"' represents 2 of clubs, and so on in order until
## chr(84) == 'T' represents K of spades.  chr(85) == 'U' is joker A and
## chr(86) == 'V' is joker B.
$D = pack('C*',33..86);

## Load the key phrase, and turn it all into uppercase
$p = shift; $p =~ y/a-z/A-Z/;

## For each letter in the key phrase, run the key setup routine (which
## is the same as the keystream routine, except that $k is set to the
## value of each successive letter in the key phrase).
$p =~ s/[A-Z]/$k=ord($&)-64,&e/eg;

## Stop setting up the key and switch to encrypting/decrypting mode.
$k = 0;

## Collect all of the alphabetic characters (in uppercase) from the input
## files (or stdin if none specified) into the variable $o
while(<>) {
    ## Change all lowercase to uppercase
    y/a-z/A-Z/;
    ## Remove any non-letters
    y/A-Z//dc;
    ## Append the input to $o
    $o .= $_;
}

## If we're encrypting, append X to the input until it's a multiple of 5 chars
if (!$d) {
    $o.='X' while length($o)%5;
}

## This next line does the crypto:
##   For each character in the input ($&), which is between 'A' and 'Z',
##   find its ASCII value (ord($&)), which is in the range 65..90,
##   subtract 13 (ord($&)-13), to get the range 52..77,
##   add (or subtract if decrypting) the next keystream byte (the output of
##     the function &e) and take the result mod 26 ((ord($&)-13+$f*&e)%26),
##     to get the range 0..25,
##   add 65 to get back the range 65..90, and determine the character with
##     that ASCII value (chr((ord($&)-13+$f*&e)%26+65)), which is between
##     'A' and 'Z'.  Replace the original character with this new one.
$o =~ s/./chr((ord($&)-13+$f*&e)%26+65)/eg;

## If we're decrypting, remove trailing X's from the newly found plaintext
$o =~ s/X*$// if $d;

## Put a space after each group of 5 characters and print the result
$o =~ s/.{5}/$& /g;
print "$o\n";

## The main program ends here.  The following are subroutines.

## The following subroutine gives the value of the nth card in the deck.
## n is passed in as an argument to this routine ($_[0]).  The A of clubs
## has value 1, ..., the K of spades has value 52, both jokers have value 53.
## The top card is the 0th card, the bottom card is the 53rd card.
sub v {
    ## The value of most cards is just the ASCII value minus 32.
    ## substr($D,$_[0]) is a string beginning with the nth card in the deck
    $v=ord(substr($D,$_[0]))-32;
    ## Special case: both jokers (53 and 54, normally) have value 53,
    ## so return 53 if the value is greater than 53, and the value otherwise.
    $v>53?53:$v;
}

## The following subroutine generates the next value in the keystream.
sub e {
    ## If the U (joker A) is at the bottom of the deck, move it to the top
    $D =~ s/(.*)U$/U$1/;
    ## Swap the U (joker A) with the card below it
    $D =~ s/U(.)/$1U/;

    ## Do the same as above, but with the V (joker B), and do it twice.
    $D =~ s/(.*)V$/V$1/; $D =~ s/V(.)/$1V/;
    $D =~ s/(.*)V$/V$1/; $D =~ s/V(.)/$1V/;

    ## Do the triple cut: swap the pieces before the first joker, and
    ## after the second joker.
    $D =~ s/(.*)([UV].*[UV])(.*)/$3$2$1/;

    ## Do the count cut: find the value of the bottom card in the deck
    $c=&v(53);
    ## Switch that many cards from the top of the deck with all but
    ## the last card.
    $D =~ s/(.{$c})(.*)(.)/$2$1$3/;

    ## If we're doing key setup, do another count cut here, with the
    ## count value being the letter value of the key character (A=1, B=2,
    ## etc.; this value will already have been stored in $k).  After the
    ## second count cut, return, so that we don't happen to do the loop
    ## at the bottom.
    if ($k) {
	$D =~ s/(.{$k})(.*)(.)/$2$1$3/;
	return;
    }

    ## Find the value of the nth card in the deck, where n is the value
    ## of the top card (be careful about off-by-one errors here)
    $c=&v(&v(0));

    ## If this wasn't a joker, return its value.  If it was a joker,
    ## just start again at the top of this subroutine.
    $c>52?&e:$c;
}
