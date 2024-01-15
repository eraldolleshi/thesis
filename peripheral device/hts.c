/** @file
 *  @brief HTS Service sample
 */

/*
 * Copyright (c) 2020 SixOctets Systems
 * Copyright (c) 2019 Aaron Tsui <aaron.tsui@outlook.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include <zephyr/kernel.h>
#include <zephyr/drivers/sensor.h>
#include <zephyr/sys/printk.h>
#include <zephyr/sys/byteorder.h>

#include <zephyr/bluetooth/bluetooth.h>
#include <zephyr/bluetooth/hci.h>
#include <zephyr/bluetooth/conn.h>
#include <zephyr/bluetooth/uuid.h>
#include <zephyr/bluetooth/gatt.h>

#ifdef CONFIG_TEMP_NRF5
static const struct device *temp_dev = DEVICE_DT_GET_ANY(nordic_nrf_temp);
#else
static const struct device *temp_dev;
#endif




static uint8_t simulate_htm;
static uint8_t indicating;
static struct bt_gatt_indicate_params ind_params;

static void htmc_ccc_cfg_changed(const struct bt_gatt_attr *attr,
				 uint16_t value)
{
	simulate_htm = (value == BT_GATT_CCC_INDICATE) ? 1 : 0;
}

static void indicate_cb(struct bt_conn *conn,
			struct bt_gatt_indicate_params *params, uint8_t err)
{
	printk("Indication %s\n", err != 0U ? "fail" : "success");
}

static void indicate_destroy(struct bt_gatt_indicate_params *params)
{
	printk("Indication complete\n");
	indicating = 0U;
}


BT_GATT_SERVICE_DEFINE(hts_svc,
	BT_GATT_PRIMARY_SERVICE(BT_UUID_HTS),
	BT_GATT_CHARACTERISTIC(BT_UUID_HTS_MEASUREMENT, BT_GATT_CHRC_INDICATE,
			       BT_GATT_PERM_NONE, NULL, NULL, NULL),
	BT_GATT_CCC(htmc_ccc_cfg_changed,
		    BT_GATT_PERM_READ | BT_GATT_PERM_WRITE),
	
);

void hts_init(void)
{
	if (temp_dev == NULL || !device_is_ready(temp_dev)) {
		printk("no device; using simulated data\n");
		temp_dev = NULL;
	} else {
		printk("device is %p, name is %s\n", temp_dev,
		       temp_dev->name);
	}
}

void hts_indicate(uint8_t str[],int n)
{
	
        uint8_t myArray[n];
        for (int i = 0; i < n; i++) {
        myArray[i] = str[i];
    }
        
        
                
        ind_params.attr = &hts_svc.attrs[2];
		ind_params.func = indicate_cb;
		ind_params.destroy = indicate_destroy;
		ind_params.data = &myArray;
		ind_params.len = n;
     	
		
		if (bt_gatt_indicate(NULL, &ind_params) == 0) {
			indicating = 1U;
		
	}
}
