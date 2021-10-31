/*
 * Tiny and fast C++ client for Open Pixel Control
 *
 * Copyright (c) 2014 Micah Elizabeth Scott <micah@scanlime.org>
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>
#include <signal.h>


class OPCClient {
public:
    OPCClient();
    ~OPCClient();

    bool resolve(const char *hostport, int defaultPort = 7890);
    bool write(const uint8_t *data, ssize_t length);
    bool write(const std::vector<uint8_t> &data);

    bool tryConnect();
    bool isConnected();

    struct Header {
        uint8_t channel;
        uint8_t command;
        uint8_t length[2];

        void init(uint8_t channel, uint8_t command, uint16_t length) {
            this->channel = channel;
            this->command = command;
            this->length[0] = length >> 8;
            this->length[1] = (uint8_t)length;
        }

        uint8_t *data() {
            return (uint8_t*) &this[1];
        }
        const uint8_t *data() const {
            return (uint8_t*) &this[1];
        }

        // Use a Header() to manipulate packet data in a std::vector
        static Header& view(std::vector<uint8_t> &data) {
            return *(Header*) &data[0];
        }
        static const Header& view(const std::vector<uint8_t> &data) {
            return *(Header*) &data[0];
        }
    };

    // Commands
    static const uint8_t SET_PIXEL_COLORS = 0;

private:
    int fd;
    struct sockaddr_in address;
    bool connectSocket();
    void closeSocket();
};


/*****************************************************************************************
 *                                   Implementation
 *****************************************************************************************/


inline OPCClient::OPCClient()
{
    fd = -1;
    memset(&address, 0, sizeof address);
}

inline OPCClient::~OPCClient()
{
    closeSocket();
}

inline void OPCClient::closeSocket()
{
    if (isConnected()) {
        close(fd);
        fd = -1;
    }
}

inline bool OPCClient::resolve(const char *hostport, int defaultPort)
{
    fd = -1;

    char *host = strdup(hostport);
    char *colon = strchr(host, ':');
    int port = defaultPort;
    bool success = false;

    if (colon) {
        *colon = '\0';
        port = strtol(colon + 1, 0, 10);
    }

    if (port) {
        struct addrinfo *addr;
        getaddrinfo(*host ? host : "localhost", 0, 0, &addr);

        for (struct addrinfo *i = addr; i; i = i->ai_next) {
            if (i->ai_family == PF_INET) {
                memcpy(&address, i->ai_addr, sizeof address);
                address.sin_port = htons(port);
                success = true;
                break;
            }
        }
        freeaddrinfo(addr);
    }

    free(host);
    return success;
}

inline bool OPCClient::isConnected()
{
    return fd > 0;
}

inline bool OPCClient::tryConnect()
{
    return isConnected() || connectSocket();
}

inline bool OPCClient::write(const uint8_t *data, ssize_t length)
{
    if (!tryConnect()) {
        return false;
    }

    while (length > 0) {
        int result = send(fd, data, length, 0);
        if (result <= 0) {
            closeSocket();
            return false;
        }
        length -= result;
        data += result;
    }

    return true;
}

inline bool OPCClient::write(const std::vector<uint8_t> &data)
{
    return write(&data[0], data.size());
}

inline bool OPCClient::connectSocket()
{
    fd = socket(PF_INET, SOCK_STREAM, IPPROTO_TCP);

    if (connect(fd, (struct sockaddr*) &address, sizeof address) < 0) {
        closeSocket();
        return false;
    }

    int flag = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (char*) &flag, sizeof flag);

    #ifdef SO_NOSIGPIPE
        flag = 1;
        setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, (char*) &flag, sizeof flag);
    #else
        signal(SIGPIPE, SIG_IGN);
    #endif

    return true;
}
