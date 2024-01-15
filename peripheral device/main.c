/* main.c - Application main entry point */

/*
 * Copyright (c) 2019 Aaron Tsui <aaron.tsui@outlook.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <zephyr/types.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/byteorder.h>
#include <zephyr/kernel.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>
#include <zephyr/bluetooth/services/bas.h>

#include "hts.h"
#define MAX_NUMBERS_PER_LINE 20
#define MAX_LINE_LENGTH 1000
static const struct bt_data ad[] = {
	BT_DATA_BYTES(BT_DATA_FLAGS, (BT_LE_AD_GENERAL | BT_LE_AD_NO_BREDR)),
	BT_DATA_BYTES(BT_DATA_UUID16_ALL,
		      BT_UUID_16_ENCODE(BT_UUID_HTS_VAL),
		     ),
};

static void connected(struct bt_conn *conn, uint8_t err)
{
	if (err) {
		printk("Connection failed (err 0x%02x)\n", err);
	} else {
		printk("Connected\n");
	}
}

static void disconnected(struct bt_conn *conn, uint8_t reason)
{
	printk("Disconnected (reason 0x%02x)\n", reason);
}

BT_CONN_CB_DEFINE(conn_callbacks) = {
	.connected = connected,
	.disconnected = disconnected,
};

static void bt_ready(void)
{
	int err;

	printk("Bluetooth initialized\n");

	hts_init();

	err = bt_le_adv_start(BT_LE_ADV_CONN_NAME, ad, ARRAY_SIZE(ad), NULL, 0);
	if (err) {
		printk("Advertising failed to start (err %d)\n", err);
		return;
	}

	printk("Advertising successfully started\n");
}

static void auth_cancel(struct bt_conn *conn)
{
	char addr[BT_ADDR_LE_STR_LEN];

	bt_addr_le_to_str(bt_conn_get_dst(conn), addr, sizeof(addr));

	printk("Pairing cancelled: %s\n", addr);
}

static struct bt_conn_auth_cb auth_cb_display = {
	.cancel = auth_cancel,
};

static void bas_notify(void)
{
	uint8_t battery_level = bt_bas_get_battery_level();

	battery_level--;

	if (!battery_level) {
		battery_level = 100U;
	}

	bt_bas_set_battery_level(battery_level);
}


// Function to add bit errors with a given Bit Error Rate (BER)
void addBitErrors(char *str, float ber) {
    int str_len = strlen(str);
    int num_errors = (int)(str_len * ber); // Calculate the number of bits to be flipped
    for (int i = 0; i < num_errors; i++) {
        // Generate a random index to flip a bit
        int random_index = rand() % str_len;
        str[random_index] ^= 1; // Flip the bit
    }
}



int main()
{ 
 int n =0;
 //Get the number of sample bytes within a Bluetooth packet
   char *number_str = getenv("NUMBER"); // Get the value of the environment variable
    if (number_str != NULL) {
        int number = atoi(number_str); // Convert string to integer
        n=number;
       }
    
    
int err;

	err = bt_enable(NULL);
	if (err) {
		printk("Bluetooth init failed (err %d)\n", err);
		return 0;
	}

	bt_ready();

	bt_conn_auth_cb_register(&auth_cb_display);


	/* Implement indicate. At the moment there is no suitable way
	 * of starting delayed work so we do it here
	 */
	k_sleep(K_SECONDS(3));
	

 
   char* file_name = getenv("INPUT_PATH");

    if (file_name == NULL) {
        perror("Environment variable INPUT_PATH is not set");
        return 1;
    }

    FILE* file = fopen(file_name, "rb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Initialize variables
    size_t buffer_size = n;
    unsigned char *buffer = (unsigned char *)malloc(buffer_size);
    size_t bytesRead;
    size_t totalBytesRead;

    if (buffer == NULL) {
        perror("Memory allocation failed");
        fclose(file);
        return 1;
    }

    // Loop until the end of the file
    while ((bytesRead = fread(buffer, sizeof(unsigned char), buffer_size, file)) > 0) {
        
        uint8_t uint8_array[MAX_NUMBERS_PER_LINE];
        if (uint8_array == NULL) {
            perror("Memory allocation failed");
            fclose(file);
            free(buffer);
            return 1;
        }
        
        int line_size =0;
        // Get the sample size 
   char *number_str = getenv("SIZE"); // Get the value of the environment variable
    if (number_str != NULL) {
        int number = atoi(number_str); // Convert string to integer
        line_size=number;
       }
        
        totalBytesRead  += bytesRead;
        
        if(totalBytesRead >= line_size && bytesRead == buffer_size){
        
        
        for (size_t i = 0; i < bytesRead - (totalBytesRead - line_size); i++) {
            uint8_array[i] = (uint8_t)buffer[i];
         printf("%d ", uint8_array[i]);
        }
        printf("\n");
        int count=0;
        count = (int) (bytesRead - (totalBytesRead - line_size));
        
        printf("%d",count);
        k_sleep(K_SECONDS(0.1));
        
        hts_indicate(uint8_array,count);
        // Move the file pointer back
        
         if (fseek(file, -(totalBytesRead - line_size), SEEK_CUR) != 0) {
        perror("Error seeking in file");
        return 1;
    }
        
        totalBytesRead = 0;
        
        }
        
        else{
        for (size_t i = 0; i < bytesRead; i++) {
            uint8_array[i] = (uint8_t)buffer[i];
         printf("%d ", uint8_array[i]);
        }
        printf("\n");
        int count=0;
        count = (int) bytesRead;
        
        printf("%d",count);
        k_sleep(K_SECONDS(0.1));
        
        hts_indicate(uint8_array,count);

       }
    }

    // Close the file and free allocated memory
    fclose(file);
    free(buffer);
    

  
	return 0;
}
