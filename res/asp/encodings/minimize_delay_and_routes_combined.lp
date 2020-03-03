%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FOR RESCHEDULING
% This file is an addition (not part of Flux)!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize a combination of delay and route penalty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% topographically sort nodes
% taken from new encoding:
%     https://github.com/potassco/train-scheduling-with-hybrid-asp/blob/master/encodings/encoding/preprocessing.lp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

node(T,(V;V'))   :- edge(T,ID,V,V').

node_count(T,V,0) :- start(T,V).
node_count(T,V',N+1) :- node_count(T,V,N), edge(T,ID,V,V').
node_pos(T,V,M) :- node(T,V), M = #max {N : node_count(T,V,N)}.

last_node(T,M)   :- train(T), M = #max{ P : node_pos(T,V,P), end(T,V) },
                    M!=#inf.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delay model along path depths instead of vertices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% TODO remove/replace delay_linear_within_one_minute.lp, cleanup constants
potlate_depth(T,D,D) :- linear_range(S), train(t), D=1..M, last_node(T,M).
potlate_depth(T,M+1,penalty_after_linear) :- train(t), last_node(T,M).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize delay first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

{ late(T,V,D,W) : visit(T,V) } :- potlate(T,V,D,W).
{ late_at_depth(T,V,D,W) : visit(T,V) } :- potlate(T,V,D,W).


next(T,V,D,D') :- potlate(T,V,D,_), potlate(T,E,D',_), D<D',
                  not potlate(T,E,D'',_) : potlate(T,E,D'',_), D''>D, D''<D'.
:- not late(T,E,D,_), late(T,E,D',_), next(T,E,D,D').

&diff{ 0-(T,V) } <= -D  :- late(T,V,D,W).
&diff{ (T,V)-0 } <=  N  :- not late(T,V,D,W), potlate(T,V,D,W),
                           N=D-1, visit(T,V).

#minimize{ W@0,T,V,D : late(T,V,D,W) }.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimze re-routing second
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

active_penalty(P,T,E) :- penalty(T,E,P), route(T,E).

#minimize{ P@0,T,E : active_penalty(P,T,E) }.
