%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% OBJECTIVE FOR RESCHEDULING
% This file is an addition (not part of Flux)!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize a combination of delay and route penalty

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimize delay first
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% late(T,D,W): train T has delay D penalized with weight W. Since delay can only increase in our setting (propagate earliest)
{ late(T,D) } :- potlate(T,D).



% not late(T,D,W) only if (T,V) - e(T,V,E) <= D-1
&diff{ (T,V)-0 } <=  N + E  :- not late(T,D), potlate(T,D),  e(T,V,E),
                           N=D-1, visit(T,V).
late(T,D) :- potlate(T,D), late(T,D+1).

#minimize{ weight_lateness_seconds@0,T,D : late(T,D), D<=upper_bound_linear_penalty }.
#minimize{ penalty_after_linear@0,T,D : late(T,D), D=upper_bound_linear_penalty }.


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% minimze re-routing second
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

active_penalty(P,T,E) :- penalty(T,E,P), route(T,E).

#minimize{ P@0,T,E : active_penalty(P,T,E) }.
