% source: https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/encodings/h3.lp
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Routing heuristic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%
% Akin to the straight-forward routing penalty minimization, the heuristic in Listing 1.7 merely tries
% to avoid routes where there is a penalty. The higher the penalty, the more those routes are to be avoided.
% In our example (Figure 1), this amounts to t1 and t2 equally shunning (2,4) and (9,11).

#heuristic route(T,E) : train(T), penalty(E,P). [P,false]
