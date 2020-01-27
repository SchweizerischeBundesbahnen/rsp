%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This is the main part of the encoding.
% It choses the routes for each train
% and also the sequence in which trains enter
% a resource, as well as, handles timing constraints.
%
% Remark: We assume that preprocessing rules out
% all routes where a connection cannot be
% realised. We do not check this constraint
% explicitly.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%% This part creates a route for every train
%%% visit(T,V)  - train T visits node V
%%% route(T,ID) - train T uses edge ID
1 { visit(T,V)      : start(T,V)    } 1 :- train(T), 2 { start(T,V) }.
visit(T,V) :- train(T), start(T,V), { start(T,V') } 1.
1 { route(T,(V,V')) : edge(T,V,V')  } 1 :- visit(T,V), not end(T,V), 2 { edge(T,V,V') }.
route(T,(V,V')) :- visit(T,V), edge(T,V,V'), { edge(T,V,V'') } 1.
visit(T,V') :- route(T,(V,V')).
route(T,(V,V')) :- bridge(T,(V,V')), edge(T,V,V'), train(T).

%%% connections
% at least one resource is the same
:- collision_free_connection(T,ID,T',ID'),
   route(T,ID), route(T',ID'),
   #sum+{1 : resource(R,ID), resource(R,ID')} 0.

%%% shared resource chunks can only be visited by one train at a time
%%% enter_resource_chunk(T,R,C,V) - train T enters resource_chunk (R,C) at node V
%%% leave_resource_chunk(T,R,C,V) - train T leaves resource_chunk (R,C) at node V
%%%                                 (edges that go into V can be part of the resource_chunk, edges leaving V cannot)
%%% seq(T,T',R,C,C')              - train T enters resource_chunk (R,C) before train T' enters resource_chunk (R,C')
%%% transition(T,V,T',V',B)       - train T arrives at node V at least B seconds before train T' arrives at node V'
enter_resource_chunk(T,R,C,V)  :- resource_chunk(T,R,C,(V,V')), route(T,(V,V')), 1 #sum {1 : edge(T,V'',V), not resource_chunk(T,R,C,(V'',V)); 1 : start(T,V)},
                                  not route(T,(VP,V)) : resource_chunk(T,R,C,(VP,V)).
leave_resource_chunk(T,R,C,V') :- resource_chunk(T,R,C,(V,V')), route(T,(V,V')), 1 #sum {1 : edge(T,V',V''), not resource_chunk(T,R,C,(V',V'')); 1 : end(T,V')},
                                  not route(T,(V',VN)) : resource_chunk(T,R,C,(V',VN)).


{ seq(T,T',R,C,C') } :- shared(T,T',R,C,C').

transition(T,V',T',U,B) :- seq(T,T',R,C,C'), shared(T,T',R,C,C'), leave_resource_chunk(T,R,C,V'), enter_resource_chunk(T',R,C',U), b(R,B).
transition(T',U',T,V,B) :- shared(T,T',R,C,C'), not seq(T,T',R,C,C'), enter_resource_chunk(T,R,C,V), leave_resource_chunk(T',R,C',U'), b(R,B).

&diff{ (T,N')-(T',M)  } <= -B :- transition(T,V',T',U,B), node_pos(T,V',N'), node_pos(T',U,M).

%%% constraints for earliest and latest arrival

% / MODIFICATION: we do not care about service intentions, remove 'relevant'
%min_e(T,N,M) :- node_pos(T,_,N), M != #sup, M = #min {E : node_pos(T,V,N), e(T,V,E), relevant(e,T,V)}.
%max_l(T,N,M) :- node_pos(T,_,N), M != #inf, M = #max {L : node_pos(T,V,N), l(T,V,L), relevant(l,T,V)}.
min_e(T,N,M) :- node_pos(T,_,N), M != #sup, M = #min {E : node_pos(T,V,N), e(T,V,E)}.
max_l(T,N,M) :- node_pos(T,_,N), M != #inf, M = #max {L : node_pos(T,V,N), l(T,V,L)}.

&diff{ 0-(T,N) } <= -E :- min_e(T,N,E).
&diff{ (T,N)-0 } <=  L :- max_l(T,N,L).

%&diff{ 0-(T,N) } <= -E    :- e(T,V,E), visit(T,V), relevant(e,T,V), node_pos(T,V,N), min_e(T,N,M), M<E.
%&diff{ (T,N)-0 } <=  L    :- l(T,V,L), visit(T,V), relevant(l,T,V), node_pos(T,V,N), max_l(T,N,M), M>L.
&diff{ 0-(T,N) } <= -E    :- e(T,V,E), visit(T,V), node_pos(T,V,N), min_e(T,N,M), M<E.
&diff{ (T,N)-0 } <=  L    :- l(T,V,L), visit(T,V), node_pos(T,V,N), max_l(T,N,M), M>L.
% \ MODIFICATION

%&diff{ 0-(T,N) } <= -E    :- e(T,V,E), visit(T,V), relevant(e,T,V), node_pos(T,V,N).
%&diff{ (T,N)-0 } <=  L    :- l(T,V,L), visit(T,V), relevant(l,T,V), node_pos(T,V,N).



%%% constraints for minimal travel time
node_bridge(T,(N,N')) :- edge(T,V,V'), node_pos(T,V,N), node_pos(T,V',N'),
                                not edge(T,U,U') : node_pos(T,U,N), node_pos(T,U',M), M != N';
                                not edge(T,U,U') : node_pos(T,U,M), node_pos(T,U',N'), M != N.
travel_time(T,(V,V'),(N,N'),D) :- edge(T,V,V'), node_pos(T,V,N), node_pos(T,V',N'),
                               D=#sum{ M,m : m(T,(V,V'),M); W,w : w(T,(V,V'),W) }.
min_travel_time(T,(N,N'),M) :- node_bridge(T,(N,N')), M = #min {D : travel_time(T,_,(N,N'),D)}.
&diff{ (T,N)-(T,N') }  <= -D :- min_travel_time(T,(N,N'),D).
&diff{ (T,N)-(T,N') }  <= -D :- route(T,(V,V')), node_pos(T,V,N), node_pos(T,V',N'),
                                travel_time(T,(V,V'),(N,N'),D),
                                D > #max {#inf; M : min_travel_time(T,(N,N'),M)}.

%%% constraints for min/max time between connections
&diff{ (T,N)-(T',M)   } <= -Min :- timed_connection(T,ID,T',ID',Min,_,V,U), route(T,ID), route(T',ID'), node_pos(T,V,N), node_pos(T',U,M).
&diff{ (T',M)-(T,N)   } <=  Max :- timed_connection(T,ID,T',ID',_,Max,V,U), route(T,ID), route(T',ID'), Max != #inf, node_pos(T,V,N), node_pos(T',U,M).
