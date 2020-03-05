% source: https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/encodings/minimize_delay.lp
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf

% delay minimization
%
% Given  thresholds Dt,v for  all  trains  and  nodes  and  the  corresponding  instances  of predicatepotlate/4,
% Listing 1.4 shows the implementation of the delay minimization. The basic idea is to use regular atoms
% to choose whether a train is delayed on its path for every potential delay (Line 1), deriving
% difference constraint atoms expressing thisinformation (lines 2–4), and ultimately using the
% regular atoms in a standard minimize statement (Line 5). In detail, for every potlate(t,v,u,w),
% a late(t,v,u,w) can be chosen if t visits v. If late(t,v,u,w) is chosen to be true, a
% difference constraint atom &diff{0-(t,v)}<=−u is derived expressing (t, v)≥u and, therefore,
% that t is delayed at v at threshold u. Otherwise,&diff{(t,v)-0}<=u−1 becomes true so that (t, v)< u holds.
% The difference constraints ensure that if the truth value of a late atom is decided, the schedule has
% to reflect this information. The minimize statement then sums up and minimizes the penalties of the late atoms
% that are true.

{ late(T,V,D,W) : visit(T,V) } :- potlate(T,V,D,W).

next(T,V,D,D') :- potlate(T,V,D,_), potlate(T,E,D',_), D<D',
                  not potlate(T,E,D'',_) : potlate(T,E,D'',_), D''>D, D''<D'.
:- not late(T,E,D,_), late(T,E,D',_), next(T,E,D,D').

&diff{ 0-(T,V) } <= -D  :- late(T,V,D,W).
&diff{ (T,V)-0 } <=  N  :- not late(T,V,D,W), potlate(T,V,D,W),
                           N=D-1, visit(T,V).

#minimize{ W,T,V,D : late(T,V,D,W) }.
