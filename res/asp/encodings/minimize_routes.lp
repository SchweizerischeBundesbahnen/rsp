% source: https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/encodings/minimize_routes.lp
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf

% minimize routes
%
% Finally, Line 6 in Listing 1.4 shows the straight forward encoding of the routing penalty minimization.
% The minimize statement merely collects the paths of the trains,sums up their penalties, and minimizes this sum.

#minimize{ P,T,E : penalty(E,P), route(T,E) }.
