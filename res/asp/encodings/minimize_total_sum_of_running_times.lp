%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PENALTY MODEL FOR SCHEDULING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% penalty model trains:  linear up to 30 seconds and then huge penalty
% TODO SIM-113 refine: not for every 30 seconds, but only at intervall 5? discuss with Erik
penalty_for_delay(1..30,1).
penalty_for_delay(31,5000).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PENALTIES FOR TRAINS TO EXCEED THEIR MINIMUM RUNNING TIME
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% pot_penalty_for_train(T,R,P): if the running time of T is R or more, then add penalty P
pot_penalty_for_train(T,M+D,P) :- train(T), minimumrunningtime(T,M), penalty_for_delay(D,P).
% act_penalty_for_train(T,R,P): is the effective running time of T more or equal R? Then add penalty P
% TODO SIM-113 choice rule does not make act_penalty_for_train safe, we get warning:
%  minimize_total_sum_of_running_times.lp:20:33-54: info: atom does not occur in any rule head: act_penalty_for_train
{act_penalty_for_train(T,M+D,P)} :- train(T), minimumrunningtime(T,M), penalty_for_delay(D,P).

% activate penalty if
% (T,V') - (T,V) >= R
&diff{ (T,V)-(T,V') } <=  -R :- act_penalty_for_train, pot_penalty_for_train(T,R,P), pot_penalty_for_train(T,R,P), train(T), start(T,V), end(T,V'), visit(T,V), visit(T,V').

% for activation of penalty: it may only remain deactivated if
% (T,V') - (T,V) < R
&diff{ (T,V')-(T,V) } <=  R-1 :- not act_penalty_for_train(T,R,P), pot_penalty_for_train(T,R,P), train(T), start(T,V), end(T,V'), visit(T,V), visit(T,V').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MINIMIZE SUM OF ALL TRAIN RUNNING TIMES <==>  MINIMIZE SUM OF HOW LONGER THEY TAKE THAN THEIR MINIMUM RUNNING TIME
% Notice that we take any solution such that
% - all agents have a green wave
% - stay within the global time window
% We do not try to minimize the global time window!
% (Design Choice Erik)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#minimize{ P,R,T: act_penalty_for_train(T,R,P) }.