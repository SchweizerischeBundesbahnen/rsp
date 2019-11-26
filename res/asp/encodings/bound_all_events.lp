%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bound for Branch and Bound
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% https://www.cs.uni-potsdam.de/~torsten/hybris.pdf
%
% Finally, to find optimal solutions, we combine the algorithms in Listing 1.3 andListing 1.8 to minimize the
% total execution time of the tasks. The adapted algorithm isgiven in Listing 1.12 . As with algorithm in 1.8,
% a propagator is registered before solving.And the control flow is similar to the branch-and-bound-based
% optimization algorithm inListing 1.3 except that we now minimize the variablebound; or better the difference
% between variable0andboundby adding the difference constraint0−bound≤btothe program in line 9 wherebis the best known
% execution time of the tasks as obtainedfrom the assignment in line 23 minus1. To bound maximum execution time of
% the task,we have to add one more line to the encoding in Listing 1.11:22&diff{ (T,M)-bound } <= -D :- duration(T,M,D).
% This makes sure that each task ends within the given bound. Running encoding andinstance with thedlpropagator results
% in the optimum bound16where the obtainedsolution corresponds to the left of the two optimal solutions indicated by
% a light bluebackground in Figure 6.

&diff{ (T,V)-bound_all_events } <= 0 :- visit(T,V).

#program bound_all_events(b).
&diff{ bound_all_events-0 } <= b.
