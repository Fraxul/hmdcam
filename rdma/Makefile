all: rdma_server rdma_client

rdma_server: tests/rdma_server.cpp RDMAConnection.cpp RDMAContext.cpp
	clang++ -g -Wall -I. -std=c++11 -o rdma_server tests/rdma_server.cpp RDMAConnection.cpp RDMAContext.cpp -libverbs -lrdmacm  -lpthread

rdma_client: tests/rdma_client.cpp RDMAConnection.cpp RDMAContext.cpp
	clang++ -g -Wall -I. -std=c++11 -o rdma_client tests/rdma_client.cpp RDMAConnection.cpp RDMAContext.cpp -libverbs -lrdmacm  -lpthread
