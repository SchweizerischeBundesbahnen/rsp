% source: https://github.com/potassco/train-scheduling-with-clingo-dl/blob/master/encodings/h2.lp
% paper: https://www.cs.uni-potsdam.de/wv/publications/DBLP_conf/lpnmr/AbelsJOSTW19.pdf

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Delay heuristic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Listing 1.6 gives a heuristic aimed at avoiding delay at earlier nodes in the paths.
% For that purpose, we first assign each node in the sub-graph of a train a natural number signifying their
% relative position (lines 1–4). Start nodes receive position 0, and from there,
% the number increases the farther a node is apart from the start nodes,indicating that they are
% visited later in the possible paths of the train. The maximum position of the end nodes is also the
% longest possible path minus one (Line 5). For apotential delay, we then select the position p and
% the maximum position m and modify the delay atom with value m−p and modifier false. This accomplishes two things.
% First, the earlier the node, the higher the value, thus delays for earlier nodes are decidedfirst.
% Second, the preferred sign of all delays is false. Intuitively, we assume that early delays are to be avoided
% since they likely lead to delays at subsequent nodes. Considering again our example in Figure 2, node 1 for
% t1 receives position 0 and node 5 position 3,respectively, while the maximum position is 5. Therefore, we receive
% values 5 and 2 for nodes 1 and 5, respectively, avoiding the delay at node 1 first, while also preferring t1 to be
% on time at both nodes.

node(T,(V;V'))   :- edge(T,V,V').
node_pos(T,V,0)  :- start(T,V).
node_pos(T,V',M+1) :- node(T,V'), not start(T,V'),
                      M = #max{ P : node_pos(T,V,P), edge(T,V,V')},
                      M!=#inf.
last_node(T,M)   :- train(T), M = #max{ P : node_pos(T,V,P), end(T,V) },
                    M!=#inf.
#heuristic late(T,V,U,W) : potlate(T,V,U,W),
                           node_pos(T,V,P),
                           last_node(T,Max). [Max-P,false]
