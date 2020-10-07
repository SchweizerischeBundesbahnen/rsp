%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DELAY MODEL for minimize_delay.lp
% This file is an addition (not part of Flux)!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% linear penalties up to upper_bound_linear_penalty and then penalty_after_linear
% penalize +1 at each time step after the scheduled time up to upper_bound_linear_penalty
% #const upper_bound_linear_penalty = .... injected from data.
% #const penalty_after_linear = .... injected from data.
% #const resolution = .... injected from data.
% #const weight_lateness_seconds = .... injected from data.

% scheduled arrival is earliest - delayatearliest
% potlate increment for arrival at or after earliest
potlate(T,V,E,weight_lateness_seconds*P) :- e(T,V,E), end(T,V), delayatearliest(T,V,P), P>0.

% potlate increments for arrival at or after [earliest+1,  (earliest - delayatarrliest) + upper_bound_linear_penalty]
linear_range(T,V,1..upper_bound_linear_penalty/resolution-P) :- train(T), end(T,V), delayatearliest(T,V,P).
potlate(T,V,E+S*resolution,weight_lateness_seconds*resolution) :- e(T,V,E), linear_range(T,V,S), end(T,V).

% potlate increment for arrival after >=  (earliest - delayatearliest) + upper_bound_linear_penalty+1
potlate(T,V,E+upper_bound_linear_penalty+1-P,penalty_after_linear) :- e(T,V,E), end(T,V),delayatearliest(T,V,P).

% TODO SIM-572 upper bound with respect to delay with respect to schedule or to earliest?
