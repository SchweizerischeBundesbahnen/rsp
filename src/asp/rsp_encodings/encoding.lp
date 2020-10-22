% copied from https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/encodings/encoding.lp
% TODO SIM-159 include as git submodule
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Encoding of path constraints.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%exactly one valid start node is chosen for each train to be visited (Line 1).
1 { visit(T,V)      : start(T,V)    } 1 :- train(T), 2 { start(SI,V) }.
visit(T,V) :- train(T), start(T,V), { start(T,V') } 1.

% From a node that is visited by a train
% and is not an end node, an edge in the relevant sub-graph is chosen as the next route
% (Line 2).
1 { route(T,(V,V')) : edge(T,V,V')  } 1 :- visit(T,V), not end(T,V), 2 { edge(T,V,V') }.
route(T,(V,V')) :- visit(T,V), edge(T,V,V'), { edge(T,V,V'') } 1.

% The new route in turn leads to a node being visited by the train (Line 3). This
% way, each train is recursively assigned a valid path. Since those paths begin at a start
% node, finish at an end node and are connected via edges valid for the respective trains,
% conditions (1) and (2) are ensured.
visit(T,V')     :- route(T,(V,V')).
route(T,(V,V')) :- bridge(T,ID,"True"), edge(T,ID,V,V').

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Encoding of conflict resolution.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% A resource conflict is possible, if two trains have an edge in their sub-graphs that is
% assigned the same resource (lines 4 and 5), and they travel through the edges around
% the same time (lines 6 and 7), more precisely, whenever the time intervals in which the
% trains may enter and leave the edges in question, extended by the time the resource is
% blocked, overlap.
shared(T,(V,V'),T',(U,U')) :- edge(T,V,V'), edge(T',U,U'), T!=T',
                              M = #max{ B : resource(R,(V,V')), resource(R,(U,U')), b(R,B) },
                              1 #sum{ 1,R : resource(R,(V,V')), resource(R,(U,U')) },
                              e(T,V,E), l(T,V',L), e(T',U,E'),
                              E <= E', E' < L+M.
% / MODIFICATION: what about the following line? Does not seem to work.
                              %, (E,T)<=(E',T').
% \ MODIFICIATION


shared(T',(U,U'),T,(V,V')) :- shared(T,(V,V'),T',(U,U')).
% / MODIFICATION: what about the following line? Does not seem to work.
                              %, e(T,V,E), e(T',U,E'), (E,T)<(E',T').
% \ MODIFICIATION
% Now, if both trains are routed through those edges a conflict occurs
% (lines 9 and 10). We resolve the conflict by making a choice which train passes through
%  their edge first (lines 11 and 12).
conflict(T,E,T',E') :- shared(T,E,T',E'), T < T',
                       route(T,E), route(T',E').
{ seq(T,E,T',E') } :-  conflict(T,E,T',E').
  seq(T',E',T,E)   :-  conflict(T,E,T',E'), not seq(T,E,T',E').
:- seq(T,(V,V'),T',(U,U')),
   route(T,(V',V'')), route(T',(U',U'')),
   shared(T,(V',V''),T',(U',U'')), not seq(T,(V',V''),T',(U',U'')).
:- seq(T,(V,V'),T',E),   route(T,(V',V'')),
   shared(T,(V',V''),T',E), not seq(T,(V',V''),T',E).
:- seq(T,E,T',(U,U')), route(T',(U',U'')),
   shared(T,E,T',(U',U'')), not seq(T,E,T',(U',U'')).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Encoding of scheduling.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% every train arrives at a node in their path
%  neither too early nor too late, respectively. Given the earliest arrival e and latest arrival l
%  of a train t at node v in their path, difference constraint atoms &difff{0-(t,v)}<= -e and &difff{(t,v)-0}<= l are derived.
% This ensures that e <= (t,v) <= l holds,
%  therefore fulfilling conditions (3) and (4).

% / MODIFICATION: we do not care about service intentions, remove 'relevant'
% &diff{ 0-(T,V) } <= -E    :- e(T,V,E), visit(T,V), relevant(e,T,V).
% &diff{ (T,V)-0 } <=  L    :- l(T,V,L), visit(T,V), relevant(l,T,V).
&diff{ 0-(T,V) } <= -E    :- e(T,V,E), visit(T,V).
&diff{ (T,V)-0 } <=  L    :- l(T,V,L), visit(T,V).
% \ MODIFICIATION



% The rule in lines 15 and 16 first calculates
% the sum d of minimal travel and waiting time for train t at edge (v,v’) in their path,
% which is the minimal difference between arrival times at nodes v and v’ for train t.
% Then, difference constraint atom &difff(t,v)-(t,v’)g<= 􀀀d is derived, which in
% turn ensures (t,v)+d  (t,v’) (Condition (5)).
&diff{ (T,V)-(T,V') }  <= -D :- route(T,(V,V')), E = (V,V'),
                                D=#sum{ M,m : m(E,M); W,w : w(T,E,W) }.

% The rule in lines 17 and 18 utilizes
%  conflict detection and resolution from Listing 1.2. Given the maximum blocked time b
%  of resources shared on (v,v’) and (u,u’), and the decision that t takes precedence
%  over t’, we derive difference constraint atom &difff(t,v’)-(t’,u)g<= 􀀀b expressing
%  linear constraint (t,v’) + b  (t0; u) for two conflicting trains t and t’ on
%  edges (v,v’) and (u,u’). Hence, t’ may only enter edge (u,u’) b seconds after t
%  has left (v,v’) (Condition (6)). Note that if several resources induce a conflict for two
%  trains on the same edges, only one difference constraint with the maximum blocked time
%  suffices since x+k  y implies x+k0  y for k  k0.
&diff{ (T,V')-(T',U) } <= -M :- seq(T,(V,V'),T',(U,U')),
                                M = #max{ B : resource(R,(V,V')), resource(R,(U,U')), b(R,B) }.

% Finally, Line 19 handles connections
% in a similar fashion. If train t on (v,v’) has a connection to t’ on (u,u’) with
% connection time w, a difference constraint atom &diff{(t,v)-(t’,u’)}<= -w is
% derived, ensuring linear constraint (t,v) + w <= (t’,u’) to hold (Condition (7)).
% This condition is required if both trains are routed through the edges (Line 20).
&diff{ (T,V)-(T',U') } <= -W :- connection(T,(V,V'),T',(U,U'),W), route(T,(V,V')), route(T',(U,U')).


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Target must be reached.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%exactly one valid end node is chosen for each train to be visited (Line 1).
% / MODIFICIATION more direct version wihtout choice
ended(T) :- visit(T,V), end(T,V).
:- not ended(T), train(T).
% \ MODIFICIATION
