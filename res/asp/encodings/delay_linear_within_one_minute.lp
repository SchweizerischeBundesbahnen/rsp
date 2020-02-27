%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DELAY MODEL for minimize_delay.lp
% This file is an addition (not part of Flux)!
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TODO SIM-171 refine
% linear penalties up to upper_bound_linear_penalty and then penalty_after_linear
% penalize +1 at each time step after the scheduled time up to upper_bound_linear_penalty
#const upper_bound_linear_penalty = 60.
#const penalty_after_linear = 5000000.
% #const weight_lateness_seconds = .... injected from data.
linear_range(1..upper_bound_linear_penalty).
potlate(T,V,E+S,weight_lateness_seconds) :- e(T,V,E), linear_range(S), end(T,V).
potlate(T,V,E+upper_bound_linear_penalty+1,penalty_after_linear) :- e(T,V,E), end(T,V).
