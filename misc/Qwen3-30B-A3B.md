Sample diff between lmdeploy and transformers:

- Message

```text
messages = [
    {"role": "user", "content": "Who are you?"}
]
```

- Diff

`compare_tensors(lmd_tensor, gt_tensor)`

```text
Loaded tensor 'input_hidden_states' from: /nvme1/zhouxinyu/0_test/gt_input_hidden_states.pt
--- Comparison for: input_hidden_states ---
  - Mean Absolute Diff: 0.00000000
  - Max Absolute Diff:  0.00000000
  - Min Absolute Diff:  0.00000000
--------------------------------------------
Loaded tensor 'cos_position_embeddings' from: /nvme1/zhouxinyu/0_test/gt_cos_position_embeddings.pt
Loaded tensor 'sin_position_embeddings' from: /nvme1/zhouxinyu/0_test/gt_sin_position_embeddings.pt
--- Comparison for: cos ---
  - Mean Absolute Diff: 0.00000000
  - Max Absolute Diff:  0.00000000
  - Min Absolute Diff:  0.00000000
----------------------------
--- Comparison for: sin ---
  - Mean Absolute Diff: 0.00000000
  - Max Absolute Diff:  0.00000000
  - Min Absolute Diff:  0.00000000
----------------------------
Loaded tensor 'hidden_states_after_layer_0' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_0.pt
--- Comparison for: hidden_states_0 ---
  - Mean Absolute Diff: 0.00005318
  - Max Absolute Diff:  0.00781250
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_1' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_1.pt
--- Comparison for: hidden_states_1 ---
  - Mean Absolute Diff: 0.00025819
  - Max Absolute Diff:  1.00000000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_2' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_2.pt
--- Comparison for: hidden_states_2 ---
  - Mean Absolute Diff: 0.00036640
  - Max Absolute Diff:  0.06250000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_3' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_3.pt
--- Comparison for: hidden_states_3 ---
  - Mean Absolute Diff: 0.00094983
  - Max Absolute Diff:  0.06250000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_4' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_4.pt
--- Comparison for: hidden_states_4 ---
  - Mean Absolute Diff: 0.00203719
  - Max Absolute Diff:  0.06250000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_5' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_5.pt
--- Comparison for: hidden_states_5 ---
  - Mean Absolute Diff: 0.00239593
  - Max Absolute Diff:  0.12500000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_6' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_6.pt
--- Comparison for: hidden_states_6 ---
  - Mean Absolute Diff: 0.00243221
  - Max Absolute Diff:  0.12500000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_7' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_7.pt
--- Comparison for: hidden_states_7 ---
  - Mean Absolute Diff: 0.00305819
  - Max Absolute Diff:  0.12500000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_8' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_8.pt
--- Comparison for: hidden_states_8 ---
  - Mean Absolute Diff: 0.00378990
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_9' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_9.pt
--- Comparison for: hidden_states_9 ---
  - Mean Absolute Diff: 0.00389295
  - Max Absolute Diff:  0.21875000
  - Min Absolute Diff:  0.00000000
----------------------------------------
Loaded tensor 'hidden_states_after_layer_10' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_10.pt
--- Comparison for: hidden_states_10 ---
  - Mean Absolute Diff: 0.00390781
  - Max Absolute Diff:  0.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_11' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_11.pt
--- Comparison for: hidden_states_11 ---
  - Mean Absolute Diff: 0.00410403
  - Max Absolute Diff:  0.21875000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_12' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_12.pt
--- Comparison for: hidden_states_12 ---
  - Mean Absolute Diff: 0.00475274
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_13' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_13.pt
--- Comparison for: hidden_states_13 ---
  - Mean Absolute Diff: 0.00509864
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_14' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_14.pt
--- Comparison for: hidden_states_14 ---
  - Mean Absolute Diff: 0.00518608
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_15' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_15.pt
--- Comparison for: hidden_states_15 ---
  - Mean Absolute Diff: 0.00572793
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_16' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_16.pt
--- Comparison for: hidden_states_16 ---
  - Mean Absolute Diff: 0.00577403
  - Max Absolute Diff:  0.18750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_17' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_17.pt
--- Comparison for: hidden_states_17 ---
  - Mean Absolute Diff: 0.00580693
  - Max Absolute Diff:  0.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_18' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_18.pt
--- Comparison for: hidden_states_18 ---
  - Mean Absolute Diff: 0.00649938
  - Max Absolute Diff:  0.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_19' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_19.pt
--- Comparison for: hidden_states_19 ---
  - Mean Absolute Diff: 0.00732974
  - Max Absolute Diff:  0.34375000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_20' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_20.pt
--- Comparison for: hidden_states_20 ---
  - Mean Absolute Diff: 0.00765385
  - Max Absolute Diff:  0.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_21' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_21.pt
--- Comparison for: hidden_states_21 ---
  - Mean Absolute Diff: 0.00985213
  - Max Absolute Diff:  0.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_22' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_22.pt
--- Comparison for: hidden_states_22 ---
  - Mean Absolute Diff: 0.01110047
  - Max Absolute Diff:  0.37500000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_23' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_23.pt
--- Comparison for: hidden_states_23 ---
  - Mean Absolute Diff: 0.01241654
  - Max Absolute Diff:  0.34375000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_24' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_24.pt
--- Comparison for: hidden_states_24 ---
  - Mean Absolute Diff: 0.01371731
  - Max Absolute Diff:  0.31250000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_25' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_25.pt
--- Comparison for: hidden_states_25 ---
  - Mean Absolute Diff: 0.01459089
  - Max Absolute Diff:  0.43750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_26' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_26.pt
--- Comparison for: hidden_states_26 ---
  - Mean Absolute Diff: 0.01496595
  - Max Absolute Diff:  0.43750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_27' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_27.pt
--- Comparison for: hidden_states_27 ---
  - Mean Absolute Diff: 0.01632025
  - Max Absolute Diff:  0.37500000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_28' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_28.pt
--- Comparison for: hidden_states_28 ---
  - Mean Absolute Diff: 0.01629196
  - Max Absolute Diff:  0.51562500
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_29' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_29.pt
--- Comparison for: hidden_states_29 ---
  - Mean Absolute Diff: 0.01631540
  - Max Absolute Diff:  0.84375000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_30' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_30.pt
--- Comparison for: hidden_states_30 ---
  - Mean Absolute Diff: 0.01751953
  - Max Absolute Diff:  0.96875000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_31' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_31.pt
--- Comparison for: hidden_states_31 ---
  - Mean Absolute Diff: 0.01891678
  - Max Absolute Diff:  0.79687500
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_32' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_32.pt
--- Comparison for: hidden_states_32 ---
  - Mean Absolute Diff: 0.02043881
  - Max Absolute Diff:  0.63281250
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_33' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_33.pt
--- Comparison for: hidden_states_33 ---
  - Mean Absolute Diff: 0.02220886
  - Max Absolute Diff:  0.50000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_34' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_34.pt
--- Comparison for: hidden_states_34 ---
  - Mean Absolute Diff: 0.02424501
  - Max Absolute Diff:  0.44335938
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_35' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_35.pt
--- Comparison for: hidden_states_35 ---
  - Mean Absolute Diff: 0.02624919
  - Max Absolute Diff:  0.44335938
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_36' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_36.pt
--- Comparison for: hidden_states_36 ---
  - Mean Absolute Diff: 0.02713859
  - Max Absolute Diff:  0.75390625
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_37' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_37.pt
--- Comparison for: hidden_states_37 ---
  - Mean Absolute Diff: 0.03224991
  - Max Absolute Diff:  1.62500000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_38' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_38.pt
--- Comparison for: hidden_states_38 ---
  - Mean Absolute Diff: 0.03599870
  - Max Absolute Diff:  1.91406250
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_39' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_39.pt
--- Comparison for: hidden_states_39 ---
  - Mean Absolute Diff: 0.04243435
  - Max Absolute Diff:  2.13671875
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_40' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_40.pt
--- Comparison for: hidden_states_40 ---
  - Mean Absolute Diff: 0.04967784
  - Max Absolute Diff:  1.53125000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_41' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_41.pt
--- Comparison for: hidden_states_41 ---
  - Mean Absolute Diff: 0.05853404
  - Max Absolute Diff:  2.43750000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_42' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_42.pt
--- Comparison for: hidden_states_42 ---
  - Mean Absolute Diff: 0.06936238
  - Max Absolute Diff:  3.62500000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_43' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_43.pt
--- Comparison for: hidden_states_43 ---
  - Mean Absolute Diff: 0.07750859
  - Max Absolute Diff:  4.37500000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_44' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_44.pt
--- Comparison for: hidden_states_44 ---
  - Mean Absolute Diff: 0.08680934
  - Max Absolute Diff:  3.25000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_45' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_45.pt
--- Comparison for: hidden_states_45 ---
  - Mean Absolute Diff: 0.09656554
  - Max Absolute Diff:  2.75000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_46' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_46.pt
--- Comparison for: hidden_states_46 ---
  - Mean Absolute Diff: 0.10573703
  - Max Absolute Diff:  8.50000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_after_layer_47' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_layer_47.pt
--- Comparison for: hidden_states_47 ---
  - Mean Absolute Diff: 0.12505163
  - Max Absolute Diff:  4.50000000
  - Min Absolute Diff:  0.00000000
-----------------------------------------
Loaded tensor 'hidden_states_before_norm' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_before_norm.pt
--- Comparison for: hidden_states_before_norm ---
  - Mean Absolute Diff: 0.12505163
  - Max Absolute Diff:  4.50000000
  - Min Absolute Diff:  0.00000000
--------------------------------------------------
Loaded tensor 'hidden_states_after_norm' from: /nvme1/zhouxinyu/0_test/gt_hidden_states_after_norm.pt
--- Comparison for: hidden_states_after_norm ---
  - Mean Absolute Diff: 0.20141853
  - Max Absolute Diff:  9.00000000
  - Min Absolute Diff:  0.00000000
-------------------------------------------------
```
