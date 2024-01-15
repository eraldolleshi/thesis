cmake_minimum_required(VERSION 3.26)
project(src C)

set(CMAKE_C_STANDARD 11)

include_directories(.)

add_executable(src
        modem_if.h
        p2G4_args.c
        p2G4_args.h
        p2G4_channel_and_modem.c
        p2G4_channel_and_modem.h
        p2G4_channel_and_modem_priv.h
        p2G4_com.c
        p2G4_com.h
        p2G4_dump.c
        p2G4_dump.h
        p2G4_func_queue.c
        p2G4_func_queue.h
        p2G4_main.c
        p2G4_pending_tx_list.c
        p2G4_pending_tx_rx_list.h
        p2G4_v1_v2_remap.c
        p2G4_v1_v2_remap.h)
