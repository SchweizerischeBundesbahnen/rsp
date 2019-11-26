% source: https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/h1.lp
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Sequence heuristic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% The heuristic in Listing 1.5 attempts to order conflicting trains by their possible arrival times at the edges
% where the conflict is located. In essence, we analyze how the time intervals of the trains are situated and
% prefer their sequence accordingly. Line 1 derives those intervals by collecting the earliest and latest time a train
% might be at an edge. Given two trains t and t’ with intervals [e, l] and [e′, l′] at the conflicting edges,
% respectively, we calculates=e′−e−(l−l′)to determine whether t should be scheduled before t’.
% If s is positive, the preferred sign of the sequence atom is also positive, thus preferring it to go
% before t’, if it is negative, the opposite is expressed.
% In detail,e′−e is positive if t’ may arrive later thantthus making itmore likely that tcan go first
% without delaying t’.
% Similarly, if l−l′is negative,t’may leave later, suggestingtto go first.
% If the results of both expressions have the samesign, one interval is contained in the other and
% if the difference is positive, the center ofthe interval oftis located earlier than the center of the interval of t’.
% For example, in Figure 2, we see thatt1andt2share a resource in(1,4)and(3,4)and the time intervals in which
% they potentially arrive at those edges are[0,7]and[4,10], respectively. Due to 4−0−(7−10) = 7, we prefer t1 to be
% scheduled before t2, which in the example clearly is the correct decision, sincet1precedest2without delaying t2.

range(T,(V,V'),E,L) :- edge(T,V,V'),e(T,V,E), l(T,V',L).
#heuristic seq(T,E,T',E') : shared(T,E,T',E'),
                            range(T,E,L,U),
                            range(T',E',L',U'). [L'-L - (U-U'),sign]


