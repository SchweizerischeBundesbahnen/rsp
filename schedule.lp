% data
train(0..100).
step(0..200).


trainpath(0,0,0).
trainpath(0,1,1).

trainpath(1,0,0).
trainpath(1,1,1).

mrt(0,1).
mrt(1,1).

resource(0..1).



% generate
{ start(T,S) } :- train(T), step(S).
{ occupied(T,S,R) } :- train(T), step(S), resource(R).

% test
1 { start(T,S): step(S) } 1 :- train(T).
:- occupied(T1,S,R), occupied(T2,S,R), T1!=T2, resource(R), step(S).


occupied(T,S+I,R) :- trainpath(T,I,R), start(T,S).
