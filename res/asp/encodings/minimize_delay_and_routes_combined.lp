%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FOR RESCHEDULING
% This file is an addition (not part of Flux)!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize a combination of delay and route penalty




{ late(T,V,D,W) : visit(T,V) } :- potlate(T,V,D,W).

next(T,V,D,D') :- potlate(T,V,D,_), potlate(T,E,D',_), D<D',
                  not potlate(T,E,D'',_) : potlate(T,E,D'',_), D''>D, D''<D'.
:- not late(T,E,D,_), late(T,E,D',_), next(T,E,D,D').

&diff{ 0-(T,V) } <= -D  :- late(T,V,D,W).
&diff{ (T,V)-0 } <=  N  :- not late(T,V,D,W), potlate(T,V,D,W),
                           N=D-1, visit(T,V).

#minimize{ W@0,T,V,D : late(T,V,D,W) }.

% TODO SIM-190 linear combination, currently hierarchical with lower prio to route section priorities
#minimize{ P@1,T,E : penalty(E,P), route(T,E) }.
