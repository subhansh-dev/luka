"""Entry point: python -m luka_gateway"""
from luka_gateway.server import GatewayServer

if __name__ == '__main__':
    server = GatewayServer()
    server.start()
