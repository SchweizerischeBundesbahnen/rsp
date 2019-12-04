#const bound_all_events = 33.
#const upper_bound_linear_penalty = 60.
#const penalty_after_linear = 5000000.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m((1,2),2).
m((2,3),2).
m((4,2),2).

resource(r12,(1,2)).
%resource(r12,(2,1)).
b(r12,0).

resource(r23,(2,3)).
%resource(r23,(3,2)).
b(r23,0).


%resource(r24,(2,4)).
resource(r24,(4,2)).
b(r24,0).



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train t1 runs 1 -> 2 -> 3 slower, earlier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% schedule is
%  at 1: 0
%  at 2: 4
%  at 3: 8
train(t1).
start(t1,1).
end(t1,3).

e(t1,1,0),
l(t1,1,bound_all_events).

edge(t1,1,2).
e(t1,2,4).
l(t1,2,bound_all_events).
w(t1,(1,2),2).

edge(t1,2,3).
e(t1,3,8).
l(t1,3,bound_all_events).
w(t1,(2,3),2).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% train t2 runs 4 -> 2 -> 3 faster, later
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% schedule is
%  at 4: 6
%  at 2: 8
%  at 3: 10
train(t2).
start(t2,4).
end(t2,3).

e(t2,4,6),
l(t2,4,bound_all_events).

edge(t2,4,2).
e(t2,2,8).
l(t2,2,bound_all_events).
w(t2,(4,2),0).

edge(t2,2,3).
e(t2,3,10).
l(t2,3,bound_all_events).
w(t2,(2,3),0).




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% introduce malfunction and adapt potlate
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% delay of 3 minutes of t1 at node 2:
&diff { 0 - (t1,2) } <= -7.

% linear penalties up to upper_bound_linear_penalty and then penalty_after_linear
% penalize +1 at each time step after the scheduled time up to upper_bound_linear_penalty
linear_range(1..upper_bound_linear_penalty).
potlate(T,V,E+S,1) :- e(T,V,E), linear_range(S), end(T,V).
potlate(T,V,E+upper_bound_linear_penalty+1,penalty_after_linear) :- e(T,V,E), end(T,V).




% delay minimization should enforce taking the faster train first:
%'dl((t1,1),0)',
%'dl((t1,2),10)',
%'dl((t1,3),14)',
%'dl((t2,4),0)',
%'dl((t2,2),8)',
%'dl((t2,3),10)'




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% dirty work around to silence ASP complaining "info: atom does not occur in any rule head"
% (we don't use all features in encoding.lp)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
bridge(0,0,0).
edge(0,0,0,0).
relevant(0,0,0).
connection(0,(0,0),0,(0,0),0).
penalty(0,0).


