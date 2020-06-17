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
linear_range(1..upper_bound_linear_penalty/resolution ).
potlate(T,V,E+S*resolution ,weight_lateness_seconds*resolution ) :- e(T,V,E), linear_range(S), end(T,V).
potlate(T,V,E+upper_bound_linear_penalty+1,penalty_after_linear) :- e(T,V,E), end(T,V).
