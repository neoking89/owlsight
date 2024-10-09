def print_logo():
    logo = """
ZZZmmmwwwwqqqqqppppdddddddbbbbkkhkkbka#*Z|!pohhhhhaahhhhaaaa
ZZZmZmmwwwwqqqqpppppppdbkkhhhkkkkho#owv_:^:qohhhaahhaahaaaaa
OOOOOZZZmwwwwqqppqpbaoabpwmwqph*#hQf<. >:">khhhhhaahhhaaaaaa
OOOmpqqwmwwmwwwqdhhmc1>,'.  `;<((;  ;)Zq:'f#khhhhhhhhhhhaaaa
OOqv]nXLZqdbkkkhqr>' ^<{jucz-    !xpM8W~ ;dahhhhhhhahhhhhhhh
OOwZ      ^:!~[['  <LoWM*WMn" [O#&#hkoWZtX#khhhhhhhhhhhhhhhh
OOOpO. ,){~l^   `^. .l(Lab_ >k8J_^ ..'I/m%@ahhhkhhhkhhhahhhh
0OO0qC, :jULh8&#oahdU?  ~> ]$Q^ ixOZQz}' <Q$#hhkkhhhhhhhhhhh
000O0qp[  c&qti' .^<nd*u  ,@X `U&kjQ$$@#1.^Y$ohkkkkhhhhhhhhh
QQ0OZZmkYph<  ~jcr(!  +d#:z8. c8z  ($$h0%?"l#%khkhhhhhhhhhhh
QQ0OOZZ0%p  1boJY8%Mp]  Y$BJ ,dMl ' >[ !8U`:m$bkhhhhhhhhhkhh
QQ000OOd8: {aL, ,@$$dW{ `!.  .z%z      U%n`;*%bkkkhkhhhhhkkk
QQQ0000#d  Xh~   l(i.hm  lrm/ ,U8p/--jk&X" u$okkkkhkkkkkhhkk
LQ000O0o#  f*u      <oC  p8/BX  iuZppZxl .z$WMhkkkkkbkkkkkkk
LQQ00OOm$( 'X*L~^`!r*pi i$/ !ZorI      !v#a?!vh*kkkkkkkkkkkk
LLQQQ0O0bB{  -UppqpZ/  l*L    +M@*pqwpahL{..  <U*kbkkkkbkkkk
CLLQLQ0O0dB0+        "n8Wi  .,!bkM&Z)~,.  <bn^ "(hhkkkbkkkkk
CLLLQQ0OZOZ&*wZXjtjUb%$#a#X` 'X**Z_  :x_ "m*Mb_ ']hkbbbdbkkk
CLLLQQQ0OZqz  I-fnx(!,i)XZ#M|iaht  lYh#I ?*bdhW) '|odbbbbkkk
CCCCLLQ00Zm" 'xi    i:.   ljd#*[  (kahm' (*dkbbW>'"Zhbbbbbbb
JCCCLLLQQwf  jpdQl `n*pQr+   }{  c*bdhd` ?odbbdoX 'rodbbbbbb
JCCCCLLLLq~ 'QQLZb[  fkpkhwv<   zopdbd#< ,qkdddkw' {odbddbbb
JCCCCLLLLwl ,0LQQ0k! .QwZZwbh0i/oqbhhdoY .1#dddkO. }apdddddb
UJJJCCLLLq> "0JCLLmJ  1dmpbdwbhdpdoXirk*[ 'j*hpav  xapdbbddd
UUJJJJCLCqt  vLJCLQq, >pqC<[ppdaahoJ.ibba+  1wao< "wbddddddd
UUUUJJJCCQZ' ILLCLLm: ipmw; CoOj1|qa~,Qhpaj' ^(t  fapddddddd
UUUUUJJCLCmz  _ULL0L. -pObY {oqi;[mdQQqpqwhm{'   +kpppdppppp
YUUUUJJJJLJwf  in0q}  um0Owvcmqqpbqwpqwwwwmk#} 'Ipbpqqpppppp
YYYUUUUUUJCJmv  `)x  >ZLQQLmmOOZZZmwwwmmwdhQ~ ` xaqpqqqqpppp
XXYYUUUUUJCCJZL>    :0mO0QQQQ0OOOZZZmmwbkO{  . ^Zpqqqqqqqppp
XXXYYYYUUJJJCJLZc>  :[fYOwqwwmmmwqqpbdmz-  ,!  }hwqqwqqqqqqq
zXXXYYYYUUUJJJJJQZX~   .l?|nXUCCJUcj1i   ;f0! 'Qqwwwwwwwqqqq
zXXXXYYUUUUJJJJJJJQw|                 l1z0p]  /bmwwwwwwqwwwq
zzzzXXXYUUUUJJJJJCCCmX" '])({-++?{/nzC00LO|  !pmmwwwwwwwwwww
zzzzXXXXYYYYUJJJJCCLCOZ[  i)X0QCCLQQLCJCZf  ^0wmmmmmmwwwwwww
czzzzXXXYYYYUUJJJJCCLLLwX,  I{cJUUUUUULZt  'OkpqwwmmmZmmmwmw
cccczzXXXXXYUUUJJJJCCLLCOmt.  ,]fzCLCQ0}  "OU}|0wmmZmZZmmmmm
vcccczzXXXXYUUUUUJJJCCLLLCZ0/,   I}jCXi  <Zb!  upZZZZZZZmmmm
vvvcczzzzXXXYYUUUUUJJCCCLLL0qwc]`   ^.  |wZZOuJmZZZZZZZZZmZZ
vvvvczzzzzXXXXYYUUUUJJJCCL0ZZOZm0X|<. IzmQ0OOwZOOOOOZZZZZZZZ
uuvvvcczzzzXXXXXXYYYUJJJJJCLLLLLL0ZZQXQ0LQ000QQ0OOOOOOZZZOZZ
uvuvvccczzzXXXXXXXXYYUUUUJJJJCCLLLCCLQLLLLQQQ000000OOOOOOOOO
uuvvvvvvcczzzXXXXXXYYYYUUUJJJCCCCCCCCLLLLLLLQQQQ000000OOOO0O
""".strip()
    print(logo)
