train(t1).
edge(t1,1,4).
start(t1,1).
end(t1,4).

e(t1,1,0).
l(t1,1,6).
relevant(e,t1,1).
relevant(l,t1,1).


e(t1,4,0).
l(t1,4,6).
relevant(e,t1,4).
relevant(l,t1,4).

m((1,4),2).
w(t1,(1,4),0).

&diff{ (t1,1)-0 } <= 2.
&diff{ 0-(t1,1) }  <= -2.
