% train t1 runs 1 -> 4
train(t1).
edge(t1,(t1,1),(t1,4)).
start(t1,(t1,1)).
end(t1,(t1,4)).
e(t1,(t1,1),0).
l(t1,(t1,1),10).
e(t1,(t1,4),0).
l(t1,(t1,4),10).
m(((t1,1),(t1,4)),2).
w(t1,((t1,1),(t1,4)),0).


% train t2 runs opposite 4 -> 1
train(t2).
edge(t2,(t2,4),(t2,1)).
start(t2,(t2,4)).
end(t2,(t2,1)).
e(t2,(t2,1),0).
l(t2,(t2,1),6).
e(t2,(t2,4),0).
l(t2,(t2,4),6).
m(((t2,4),(t2,1)),1).
w(t2,((t2,4),(t2,1)),0).

% common resource r1 without release time 1
resource(r,((t1,1),(t1,4))).
resource(r,((t2,4),(t2,1))).
b(r,5).



% dirty work around to silence ASP complaining "info: atom does not occur in any rule head"
% (we don't use all features in encoding.lp)
bridge(0,0,0).
edge(0,0,0,0).
relevant(0,0,0).
m(0,1).
connection(0,(0,0),0,(0,0),0).
penalty(0,0).
