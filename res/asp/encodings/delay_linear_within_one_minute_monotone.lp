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
% potlate(T,D,W): delay larger or equal S is penalized with weight W for train T
%TODO alpha/beta weights
% delay at least S is penalized with 1 up to upper_bound_linear_penalty (inclusive)
potlate(T,S) :- S=1..upper_bound_linear_penalty, train(T).
% delay larger than upper_bound_linear_penalty is penalized with penalty_after_linear
potlate(T,upper_bound_linear_penalty+1) :- train(T).

weight(S,weight_lateness_seconds) :- S=1..upper_bound_linear_penalty.
% delay larger than upper_bound_linear_penalty is penalized with penalty_after_linear
weight(upper_bound_linear_penalty+1,penalty_after_linear).
