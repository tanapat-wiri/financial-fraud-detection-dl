<div class="container-fluid main-container">

<div id="header">

# Financial Fraud Detection and Transaction Clustering

#### Thanapat_wiri


</div>

<div id="section" class="section level1 tabset">

# 

<div id="neural-networks" class="section level2">

## Neural Networks

<div id="setup" class="section level3">

### setup

``` r
library(torch)
library(luz)
library(dplyr)
```

    ## 
    ## Attaching package: 'dplyr'

    ## The following objects are masked from 'package:stats':
    ## 
    ##     filter, lag

    ## The following objects are masked from 'package:base':
    ## 
    ##     intersect, setdiff, setequal, union

``` r
credit_data <- read.csv("creditcard_2023.csv")
x <- credit_data[, c(paste0("V", 1:28), "Amount")]
y <- credit_data$Class

x$Amount <- scale(x$Amount)

# แบ่งข้อมูล Train/Test
set.seed(123)
trainIndex <- sample(1:nrow(x), 0.8 * nrow(x))
x_train <- x[trainIndex, ]
y_train <- y[trainIndex]
x_test <- x[-trainIndex, ]
y_test <- y[-trainIndex]

# แปลงข้อมูลเป็น Torch Tensor
library(torch)
x_train_tensor <- torch_tensor(as.matrix(x_train), dtype = torch_float())
y_train_tensor <- torch_tensor(as.matrix(y_train), dtype = torch_float())
x_test_tensor <- torch_tensor(as.matrix(x_test), dtype = torch_float())
y_test_tensor <- torch_tensor(as.matrix(y_test), dtype = torch_float())
```

``` r
# สร้างโมเดล Neural Network
nn_model <- nn_module(
  initialize = function() {
    self$fc1 <- nn_linear(ncol(x_train), 64)  # Hidden Layer 1
    self$fc2 <- nn_linear(64, 32)            # Hidden Layer 2
    self$output <- nn_linear(32, 1)          # Output Layer
    self$sigmoid <- nn_sigmoid()             # Activation Function for Output
  },
  forward = function(x) {
    x %>% 
      self$fc1() %>% nnf_relu() %>%
      self$fc2() %>% nnf_relu() %>%
      self$output() %>% self$sigmoid()
  }
)

# กำหนด Optimizer และ Loss Function
model <- nn_model()
optimizer <- optim_adam(model$parameters, lr = 0.001)
loss_fn <- nnf_binary_cross_entropy
model
```

    ## An `nn_module` containing 4,033 parameters.
    ## 
    ## ── Modules ─────────────────────────────────────────────────────────────────────
    ## • fc1: <nn_linear> #1,920 parameters
    ## • fc2: <nn_linear> #2,080 parameters
    ## • output: <nn_linear> #33 parameters
    ## • sigmoid: <nn_sigmoid> #0 parameters

``` r
# ฝึกโมเดล
epochs <- 120
for (epoch in 1:epochs) {
  optimizer$zero_grad()
  y_pred <- model(x_train_tensor)
  loss <- loss_fn(y_pred, y_train_tensor)
  loss$backward()
  optimizer$step()
  cat(sprintf("Epoch %d, Loss: %f\n", epoch, loss$item()))
}
```

    ## Epoch 1, Loss: 0.702382
    ## Epoch 2, Loss: 0.694410
    ## Epoch 3, Loss: 0.686748
    ## Epoch 4, Loss: 0.679320
    ## Epoch 5, Loss: 0.672051
    ## Epoch 6, Loss: 0.664912
    ## Epoch 7, Loss: 0.657807
    ## Epoch 8, Loss: 0.650643
    ## Epoch 9, Loss: 0.643337
    ## Epoch 10, Loss: 0.635851
    ## Epoch 11, Loss: 0.628157
    ## Epoch 12, Loss: 0.620227
    ## Epoch 13, Loss: 0.612044
    ## Epoch 14, Loss: 0.603585
    ## Epoch 15, Loss: 0.594828
    ## Epoch 16, Loss: 0.585760
    ## Epoch 17, Loss: 0.576377
    ## Epoch 18, Loss: 0.566677
    ## Epoch 19, Loss: 0.556645
    ## Epoch 20, Loss: 0.546300
    ## Epoch 21, Loss: 0.535673
    ## Epoch 22, Loss: 0.524790
    ## Epoch 23, Loss: 0.513680
    ## Epoch 24, Loss: 0.502369
    ## Epoch 25, Loss: 0.490880
    ## Epoch 26, Loss: 0.479255
    ## Epoch 27, Loss: 0.467533
    ## Epoch 28, Loss: 0.455753
    ## Epoch 29, Loss: 0.443954
    ## Epoch 30, Loss: 0.432177
    ## Epoch 31, Loss: 0.420456
    ## Epoch 32, Loss: 0.408827
    ## Epoch 33, Loss: 0.397326
    ## Epoch 34, Loss: 0.385984
    ## Epoch 35, Loss: 0.374831
    ## Epoch 36, Loss: 0.363894
    ## Epoch 37, Loss: 0.353198
    ## Epoch 38, Loss: 0.342765
    ## Epoch 39, Loss: 0.332614
    ## Epoch 40, Loss: 0.322765
    ## Epoch 41, Loss: 0.313233
    ## Epoch 42, Loss: 0.304032
    ## Epoch 43, Loss: 0.295173
    ## Epoch 44, Loss: 0.286661
    ## Epoch 45, Loss: 0.278501
    ## Epoch 46, Loss: 0.270694
    ## Epoch 47, Loss: 0.263237
    ## Epoch 48, Loss: 0.256126
    ## Epoch 49, Loss: 0.249357
    ## Epoch 50, Loss: 0.242923
    ## Epoch 51, Loss: 0.236814
    ## Epoch 52, Loss: 0.231018
    ## Epoch 53, Loss: 0.225525
    ## Epoch 54, Loss: 0.220321
    ## Epoch 55, Loss: 0.215394
    ## Epoch 56, Loss: 0.210727
    ## Epoch 57, Loss: 0.206307
    ## Epoch 58, Loss: 0.202118
    ## Epoch 59, Loss: 0.198145
    ## Epoch 60, Loss: 0.194375
    ## Epoch 61, Loss: 0.190795
    ## Epoch 62, Loss: 0.187392
    ## Epoch 63, Loss: 0.184159
    ## Epoch 64, Loss: 0.181087
    ## Epoch 65, Loss: 0.178167
    ## Epoch 66, Loss: 0.175393
    ## Epoch 67, Loss: 0.172756
    ## Epoch 68, Loss: 0.170248
    ## Epoch 69, Loss: 0.167862
    ## Epoch 70, Loss: 0.165590
    ## Epoch 71, Loss: 0.163423
    ## Epoch 72, Loss: 0.161355
    ## Epoch 73, Loss: 0.159380
    ## Epoch 74, Loss: 0.157491
    ## Epoch 75, Loss: 0.155682
    ## Epoch 76, Loss: 0.153951
    ## Epoch 77, Loss: 0.152291
    ## Epoch 78, Loss: 0.150700
    ## Epoch 79, Loss: 0.149172
    ## Epoch 80, Loss: 0.147706
    ## Epoch 81, Loss: 0.146297
    ## Epoch 82, Loss: 0.144942
    ## Epoch 83, Loss: 0.143639
    ## Epoch 84, Loss: 0.142384
    ## Epoch 85, Loss: 0.141176
    ## Epoch 86, Loss: 0.140012
    ## Epoch 87, Loss: 0.138889
    ## Epoch 88, Loss: 0.137806
    ## Epoch 89, Loss: 0.136760
    ## Epoch 90, Loss: 0.135750
    ## Epoch 91, Loss: 0.134773
    ## Epoch 92, Loss: 0.133830
    ## Epoch 93, Loss: 0.132917
    ## Epoch 94, Loss: 0.132034
    ## Epoch 95, Loss: 0.131180
    ## Epoch 96, Loss: 0.130353
    ## Epoch 97, Loss: 0.129551
    ## Epoch 98, Loss: 0.128773
    ## Epoch 99, Loss: 0.128019
    ## Epoch 100, Loss: 0.127286
    ## Epoch 101, Loss: 0.126574
    ## Epoch 102, Loss: 0.125881
    ## Epoch 103, Loss: 0.125208
    ## Epoch 104, Loss: 0.124553
    ## Epoch 105, Loss: 0.123915
    ## Epoch 106, Loss: 0.123293
    ## Epoch 107, Loss: 0.122687
    ## Epoch 108, Loss: 0.122097
    ## Epoch 109, Loss: 0.121520
    ## Epoch 110, Loss: 0.120957
    ## Epoch 111, Loss: 0.120407
    ## Epoch 112, Loss: 0.119869
    ## Epoch 113, Loss: 0.119343
    ## Epoch 114, Loss: 0.118829
    ## Epoch 115, Loss: 0.118325
    ## Epoch 116, Loss: 0.117832
    ## Epoch 117, Loss: 0.117349
    ## Epoch 118, Loss: 0.116875
    ## Epoch 119, Loss: 0.116411
    ## Epoch 120, Loss: 0.115955

``` r
# ทำนายผลบน Test Data
y_pred_tensor <- model(x_test_tensor)

# แปลง Tensor เป็น Matrix
y_pred <- as.matrix(y_pred_tensor)
y_pred <- ifelse(y_pred > 0.5, 1, 0)

# ประเมินผลลัพธ์
confusionMatrix <- table(Predicted = y_pred, Actual = y_test)
print(confusionMatrix)
```

    ##          Actual
    ## Predicted     0     1
    ##         0 55906  4085
    ##         1   997 52738

``` r
# คำนวณ Metrics
accuracy <- sum(diag(confusionMatrix)) / sum(confusionMatrix)
precision <- confusionMatrix[2, 2] / sum(confusionMatrix[2, ])
recall <- confusionMatrix[2, 2] / sum(confusionMatrix[, 2])
f1_score <- 2 * (precision * recall) / (precision + recall)

cat(sprintf("Accuracy: %.2f%%\n", accuracy * 100))
```

    ## Accuracy: 95.53%

``` r
cat(sprintf("Precision: %.2f\n", precision))
```

    ## Precision: 0.98

``` r
cat(sprintf("Recall: %.2f\n", recall))
```

    ## Recall: 0.93

``` r
cat(sprintf("F1-Score: %.2f\n", f1_score))
```

    ## F1-Score: 0.95

</div>

</div>

<div id="dbscan" class="section level2">

## DBSCAN

<div id="setup-1" class="section level3">

### setup

``` r
library(dbscan)
```

    ## 
    ## Attaching package: 'dbscan'

    ## The following object is masked from 'package:stats':
    ## 
    ##     as.dendrogram

``` r
library(dplyr)
library(ggplot2)

# โหลดข้อมูล
data = read.csv("bank_transactions_data_2.csv",header=T,na.strings="?")

# ตรวจสอบโครงสร้างข้อมูล
str(data)
```

    ## 'data.frame':    2512 obs. of  16 variables:
    ##  $ TransactionID          : chr  "TX000001" "TX000002" "TX000003" "TX000004" ...
    ##  $ AccountID              : chr  "AC00128" "AC00455" "AC00019" "AC00070" ...
    ##  $ TransactionAmount      : num  14.1 376.2 126.3 184.5 13.4 ...
    ##  $ TransactionDate        : chr  "2023-04-11 16:29:14" "2023-06-27 16:44:19" "2023-07-10 18:16:08" "2023-05-05 16:32:11" ...
    ##  $ TransactionType        : chr  "Debit" "Debit" "Debit" "Debit" ...
    ##  $ Location               : chr  "San Diego" "Houston" "Mesa" "Raleigh" ...
    ##  $ DeviceID               : chr  "D000380" "D000051" "D000235" "D000187" ...
    ##  $ IP.Address             : chr  "162.198.218.92" "13.149.61.4" "215.97.143.157" "200.13.225.150" ...
    ##  $ MerchantID             : chr  "M015" "M052" "M009" "M002" ...
    ##  $ Channel                : chr  "ATM" "ATM" "Online" "Online" ...
    ##  $ CustomerAge            : int  70 68 19 26 26 18 37 67 51 55 ...
    ##  $ CustomerOccupation     : chr  "Doctor" "Doctor" "Student" "Student" ...
    ##  $ TransactionDuration    : int  81 141 56 25 198 172 139 291 86 120 ...
    ##  $ LoginAttempts          : int  1 1 1 1 1 1 1 1 1 1 ...
    ##  $ AccountBalance         : num  5112 13759 1122 8569 7429 ...
    ##  $ PreviousTransactionDate: chr  "2024-11-04 08:08:08" "2024-11-04 08:09:35" "2024-11-04 08:07:04" "2024-11-04 08:09:06" ...

``` r
# เลือกฟีเจอร์ที่สำคัญสำหรับ Clustering
features <- data %>%
  select(TransactionAmount, TransactionDuration, LoginAttempts, AccountBalance) %>%
  scale()  # Normalize ข้อมูล
```

``` r
kNNdistplot(features, k = 4)  # เลือก k = minPts - 1
```

<img
src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABUAAAAPACAMAAADDuCPrAAAA51BMVEUAAAAAADoAAGYAOjoAOmYAOpAAZmYAZrY6AAA6OgA6Ojo6OmY6OpA6ZmY6ZpA6ZrY6kJA6kLY6kNtmAABmADpmOgBmOjpmOmZmZjpmZmZmZpBmkLZmkNtmtrZmtttmtv+QOgCQOjqQZgCQZjqQZmaQkGaQkJCQkLaQtraQttuQtv+Q29uQ2/+2ZgC2Zjq2kDq2kGa2kLa2tpC2tra2ttu225C227a229u22/+2///bkDrbkGbbtmbbtpDbtrbbttvb25Db27bb29vb2//b/7bb////tmb/25D/27b/29v//7b//9v///8OiGGKAAAACXBIWXMAAB2HAAAdhwGP5fFlAAAgAElEQVR4nO3dfWPURoLgYZmQxYEdsgtHbghzgTtmd4Y7YLghdyQeyF3IYhxa3//zbEv9ppbKXd1l2Sqpn+ePxG7bapWEf9Zbq4sSgCTF0DMAMFYCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQVIJKAAiQQUIJGAAiQSUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARLlHdACoDf9J6r3KfZo6KUNTEvvjep7gn26hj8YwNESUIBEAgqQSEABEgkoQCIBBUgkoACJBBQgkYACJBJQgEQCCpBIQAESCShAIgEFSCSgAIkEFCCRgAIkElCARAIKkEhAARIJKEAiAQXYtncoBBRgm4CGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoECUgIYJKBAloGECCkQJaJiAAlECGiagQJSAhgkoECWgYQIKRAlomIACUQIaJqBAlICGCSgQJaBhAgpECWiYgAJRAhomoEDUEQT017c//nbwDwkoEDXdgH78VP139ua0mLv94sCfFlAgaqIBvXheVfNlWb4qlr49bAICCkRNM6Cf683O4uTl+fy/9x4/rj59cNAUBBSImmRAvzycB/PuN/Nt0EfziM4fmL0uFh/s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"
role="img" width="672" />

``` r
# ใช้ DBSCAN
dbscan_result <- dbscan(features, eps = 0.5, minPts = 5)

# เพิ่มผลลัพธ์การจัดกลุ่มลงในข้อมูล
data$Cluster <- dbscan_result$cluster

# ตรวจสอบจำนวนคลัสเตอร์
table(data$Cluster)
```

    ## 
    ##    0    1    2 
    ##  186 2319    7

``` r
# ตรวจสอบธุรกรรมที่ไม่เข้ากับกลุ่ม (Noise: Cluster = 0)
anomalies <- data %>% filter(Cluster == 0)

# แสดงตัวอย่างธุรกรรมผิดปกติ
head(anomalies)
```

    ##   TransactionID AccountID TransactionAmount     TransactionDate TransactionType
    ## 1      TX000024   AC00453            345.84 2023-05-02 18:25:46           Debit
    ## 2      TX000027   AC00441            246.93 2023-04-17 16:37:01           Debit
    ## 3      TX000033   AC00060            396.45 2023-09-25 16:26:00           Debit
    ## 4      TX000039   AC00478            795.31 2023-10-12 17:07:40           Debit
    ## 5      TX000062   AC00002            263.99 2023-05-16 16:07:30           Debit
    ## 6      TX000086   AC00098           1340.19 2023-09-29 17:22:10          Credit
    ##     Location DeviceID     IP.Address MerchantID Channel CustomerAge
    ## 1 Fort Worth  D000162 191.82.103.198       M083  Online          22
    ## 2      Miami  D000046 55.154.161.250       M029     ATM          23
    ## 3   New York  D000621 133.67.250.163       M007     ATM          49
    ## 4       Mesa  D000077   49.29.37.185       M048     ATM          66
    ## 5     Dallas  D000285   7.146.35.122       M087  Branch          79
    ## 6     Austin  D000574 165.114.224.47       M012  Online          54
    ##   CustomerOccupation TransactionDuration LoginAttempts AccountBalance
    ## 1            Student                 142             3        1402.50
    ## 2            Student                 158             5         673.35
    ## 3           Engineer                 168             3        9690.15
    ## 4             Doctor                  90             2        7914.88
    ## 5            Retired                 227             2        4175.02
    ## 6           Engineer                  30             1        8654.28
    ##   PreviousTransactionDate Cluster
    ## 1     2024-11-04 08:07:04       0
    ## 2     2024-11-04 08:11:38       0
    ## 3     2024-11-04 08:11:13       0
    ## 4     2024-11-04 08:11:17       0
    ## 5     2024-11-04 08:11:03       0
    ## 6     2024-11-04 08:06:53       0

``` r
# สร้างกราฟแสดงผล
ggplot(data, aes(x = TransactionAmount, y = TransactionDuration, color = factor(Cluster))) +
  geom_point() +
  labs(title = "DBSCAN Clustering of Transactions",
       x = "Transaction Amount",
       y = "Transaction Duration",
       color = "Cluster")
```

<img
src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABUAAAAPACAMAAADDuCPrAAABaFBMVEUAAAAAADoAAGYAOjoAOmYAOpAAZpAAZrYAujgzMzM6AAA6OgA6Ojo6OmY6OpA6ZmY6ZpA6ZrY6kJA6kLY6kNtNTU1NTW5NTY5Nbo5NbqtNjshhnP9mAABmADpmOgBmOjpmOpBmZgBmZjpmZmZmZpBmkJBmkLZmkNtmtrZmtttmtv9uTU1ubm5ubo5ujqtujshuq+SOTU2Obk2Obm6Oq8iOq+SOyOSOyP+QOgCQZgCQZjqQZmaQZpCQkGaQkLaQtraQttuQtv+Q29uQ2/+rbk2rbm6rjm6ryOSr5P+2ZgC2Zjq2Zma2kDq2kGa2kJC2tpC2tra2ttu229u22/+2///Ijk3Ijm7IyKvI5P/I///bkDrbkGbbtmbbtpDbtrbbttvb27bb29vb2//b/7bb///kq27kyI7kyKvk///r6+v4dm3/tmb/yI7/25D/27b/29v/5Kv/5Mj/5OT//7b//8j//9v//+T///8Ina/WAAAACXBIWXMAAB2HAAAdhwGP5fFlAAAgAElEQVR4nOy9i7ckR5HmmTWSjm6XipE4zeXRpYVe8ehF2sNAz+6glXZ6tAvL9GVgFlCrDxRSN7AwTd6iSuiIq/z398bbH5+Zm3t4hEVG2neOlJkR7va5203/VUS4R+ThZDKZTKYiHbQbYDKZTOcqA6jJZDIVygBqMplMhTKAmkwmU6EMoCaTyVQoA6jJZDIVygBqMplMhTKAmkwmU6EMoCaTyVQoA6jJZDIVygBqMplMhTKAmkwmU6EMoCaTyVQoA6jJZDIVygBqMplMhVoNoM8eHkY9evW9aceTg7PjW79z69z99681tV549b3f+8HuPup2vPaL0ObmcPh3/xTYfsGt+fbh8CZqXxzy09cPD36Q00WBqsb88FHT4P9r+Pj0EAt2dVktkTaTaaNSAei9Hnx72PHEH/KfH2vc/dgp/ndOKHfHi+97Lvej1wdmY+uOZwKgKGQmCT76mqBwTbp0TZ7CaQO0T4AB1HRB0gLoRMoAoIeX++0N6hy99Hsc6YHHiKd+0b6wswEDFIbMIsF9XEnhinTpeTkdb+sCdEyAAdR0QVoToAPH7n7zzmE6UHwyMfPjj746HVLdn4wfPteeu7ebx+PKFnYvvtec63dxHEg0o/gVb0uHxi+4JWKq4JBZJFgfG/d5e/Ae2nHfG+8qxjoybpouURoAPXln1g5AO769PJYYuXd3MxZvTtKn8//nr3vXPBuTD914PUCnoY0ASoTcOEBvvH46MoCaTGtJCaAt2Lrx7wG0OQ/tij2Jzrw7nN741zT9SaL7Sl/wR/L9/hceOqEQQImQ2wfoF+AOA6jJtJa0ANqegf6gf0MA9At+8Zf7MD43nvh0vI/pkaU/Jh23AIBSIQ2gOTKAmi5RagBtzptbAEQARQemo54Ey5ROz772jZ87le89nrpFWlvnEBMAlAo5IMEh1dSo5++8cpgWPQ2TUL3L83eaKxSvvT/VevPuxw8PDz73Ayfmm12xF741ZuWuWZf04PO/x2z8zdecoOOsF0BlAFDHvfn40Tvtaq1hvRhsiN+5E6jWtPanr7RXqcMEOCT1miz3MpnORmoAHa92htdA+9HXzCl/+xRqukaK1IHHY2Rr28C6N48BSoZkAOqsemoXPXkAnXZ+/vdDrTdvut0uQIdiA26ef3X4DAA6rUlog2YC9GbwGTzGpQagIWHnTqDa6fTroQUvue1xARo0WeplMp2R1AA6nns7AL1rptt7dLTD78VwBf143Io0DF33rL6zfTqexMcAJUMyAL05TGpg5QLUXX41Hkw/6lnjxHw0luoa2y5h7SL+z1GLpp1d8TyADu5emA5hoCFh506gmrv4zG2PA9CwyUIvk+mcpArQdsj460Cn2fB+AD549b/8zotCr218Mh0OjZfjetvxBqUYoGRIGqBNlc83rfr4J4d+53Ti2vSn2fnxjw/TRYrD4a9/f/r4d27Mw6E5t22P7Vr7ZlNzqPZhsPDqNOxsiv9mWtAlvQbquLfTZd/+fXP6PVz3jRsCOhdXawq9+F5/BOknwO2i32SRl8l0TtID6FMI0OF0715374wbX3tvqkROVYxsnKbsR9uGxi97hdx24JA0QJ2D5uGty4/BwO3hF6KY/abhALhhSfePx/Pwbip3nutu/KcgB6CTu9O29s8RNyTuHK720nh9on0XAhQ0WeJlMp2VNgdQl2Yf/3S8Q+jF94ZKJEAncDjTSIPt0/6YpxZAX/wnXNgDwU3n9WRycGIOLbyJoPw0AujEq4lmGQAd3J96YQYSxg0JOhdXc5DqnLF7AAVNlniZTGelTQB0vAba3gfkXQu7++jdVzqEDqeOJECnleXO+B5t+9ELT+FzAdreNOldWpj4cePEfzoeZzksGWL6h15uu6Krsl6jb6brsFKAvvT7qNC9x3BUGBwDgs5F1Z7GMQOAoiZneplM29cmroH6y5jiGeiPvnZwLpoR10DdSYtxkmKy7U/iq1wDHeeJPvcPvw8KB3fwhweXZExvGWXIRm+ncxYtBWhwbnz32//33YfTaXW6c1G1p/H5dgBQ1OQML5PpPKQH0Jt4Ft7d7KuZdhjOHYm5huBZGv3wnWy7k/g6s/AOJvtrCxRAG3MhQCfqLQjQ37zzytA0AqCgc1E1cMGyCKDQy2Q6H21sHeh4YBoO0f5EGyzafPK5b71/isk13VI/2LYn8aJ1oF3IxEL60cibhN42QKf1nAxA485F1aoBNPYymc5JagAdz5yDwfh0PEn0roU6azz99YLDlLt/KXMs5ti2J/GSO5H6kCxA7/Vx97hnfw4F3Wtf9xTeWYuaC9D+KsejV7/58//vdQagQefiapkAfTJeA00n0mQ6J2neC98Nc3wEGl6ZHAZkdOP6MInuz5U8cxYsjpvbm5sE98I/HbkRwi56AtLdsH7RnUQKuSYBqAtedxnWuGHOJJIzddOu5zx5k0gQam7n4mruJNK01kkwiZT0MpnOSlt7GlMz8Jpyw+ugJ87yR5eAwwLP4ErmWN2zbRaEvxIfIdIhgxnz/mw/hIMH0Cc+W4LVjqK1pfFE2pN5y5iGwE6Vp9QpPOhcXM1Z5jCUD5cxgSZLvEyms9LGngd6Gp+c1FDkpfH26I9G3sYP72zjhKs5h8+ebX8umn4eqHdWPp3iPzmMsBsfDv12AFDneHa4qiAC6LSQ/tPqC+kd/k9No45Ao84R1aa5uXaLbCF9ystkOi9pAPTjf3nH4ZgLUHcdaHfj3y9+PyxjGqDQzrZ3jwVyHh8fnlwPh5E+t59CgBIhByQ0e1/8v4fpjgF23W36zX2KbcO8B5YeXvxFf99jOE2WuL++uQ2rffx+yJJ5t3I6R6DtuXh7lykFtbhzoNp4K+d45j0mgL2VM+llMp2V9H4TyRlLvoZzwxtv63R8+aEfaRi1ARdvxmup7pWAGwhQGBJNrH/z4YjEoHRf6E1/Hj66yssA1F/HGrAxfjJHCUCDn34iGhJ1DlQLHybiJIB9mEjSy2Q6K2kBdFr1FwD0xelE3MXaa87DztxVNS+O1wGC45f+QmIA0HZUg2XzIKRzUjo8Z+7NZz1Ap4e5jef+T0cGTPfw909okwF0asNr/0cMk0/HBg4PCyhZxjTm+sH/8jZxXRJ1Lq7m/HGG9A4JiNLmPs4u7WUynZN0fhf+G87jc12AvvCav5r6o3fa558FPxd/z5r//Mj7EXewlrOfRgon/4d74iOFId3FQ92Tju8ZPgL09PFP25Z9bnre3ofNUvO/bt/+pn368OeGvggB2tk01W5QG9vfrX9h+oekaCF996jnptXc0qKoc1G1U/BAZScBzgKmoMkyL5PpjLQaQE1iEb9dbzKZtiYD6EbkHJ55i9BNJtN2ZQDdiJyruPj5SSaTaXMygG5E7WKp5vpru5DKJqRNpnOQAXQrcldt2QGoyXQWMoBuRc4vmHze+GkynYUMoNtR/xPp37Tf9zWZzkQGUJPJZCqUAdRkMpkKZQA1mUymQhlATSaTqVAGUJPJZCqUAdRkMpkKZQA1mUymQhlATSaTqVAGUJPJZCqUAdRkMpkKZQA1mUymQhlATSaTqVAGUJPJZCqUAdRkMpkKtRJAj3nKrlBbuv7WfV1/ZXttf2X70/B6FjKA4gboulv3Vf2V7bX9le0NoECFSVTTZRPkwruvba/tr2xvAAUqTKKaLpsgF959bXttf2V7AyhQYRLVdNkEufDua9tr+yvbG0CBCpOopssmyIV3X9te21/Z3gAKVJhENV02QS68+9r22v7K9gZQoMIkqumyCXLh3de21/ZXtjeAAhUmUU2XTZAL7762vba/sr0BFKgwiWq6bIJcePe17bX9le0NoECFSVTTZRPkwruvba/tr2xvAAUqTKKaLpsgF959bXttf2V7AyhQYRLVdNkEufDua9tr+yvbG0CBCpOopssmyIV3X9te21/Z3gAKVJhENV02QS68+9r22v7K9gZQoMIkqumyCXLh3de21/ZXtjeAAhUmUU2XTZAL7762vba/sr0BFKgwiWq6bIJcePe17bX9le0NoECFSVTTZRPkwruvba/tr2xvAAUqTKKaLpsgF959bXttf2V7AyhQYRLVdNkEufDua9tr+yvbG0CBCpOopssmyIV3X9te21/Z3gAKVJhENV02QS68+9r22v7K9gZQoMIkqumyCXLh3de21/ZXtjeAAhUmUU2XTZAL7762vba/sr0BFKgwiWq6bIJcePe17bX9le0NoECFSVTTZRPkwruvba/tr2xvAAUqTKKaLpsgF959bXttf2V7AyhQYRLVdNkEufDua9tr+yvbG0CBCpOopssmyIV3X9te21/Z/gIA+m9/f339xf/tj92Hz3701vX1d34JPjgqTKKaLpsgF959bXttf2X7/QP0D9etvvyr5sNfvtd++FL8wVVhEtV02QS58O5r22v7K9vvHqB/fuuL/+l0+uTvr/+m+fTB9Vd+efrkh9df+WP4wVVhElfR1b2iBqzoH0t7COl1//Ze6t3Xttf2V7bfPUA/uP5u8/Lnt5oDze7/94eeX/zH4IOnwiSuoasrQFADqIpuW2l3X9te21/ZfvcA7fWX7zW0/EN3HHr/+t3gg6fCJBYIHU9yRa+uEEENoBq67aU9hJXttf2V7S8FoH9+qzlR/+D6++2nPzXs9D54KkxivuDxJFf0ChLUAKqg21Eq9qM2QhA1f2X7CwHo/3jr+vun02c/7M/WG5x6H7pSf9VrbjOlGmiYUTSjimlRTQDVbonJJFQhQD+4vv7ifz0tCdAipmXgMOSnAVRdBlDT2akMoJ/9P//rW9df/N89gH7pV94Hv0L2YXzGqbgj4oScL2qn8HEDVFztFL6z1/ZXtr+QU/jTvzXn8Kkj0EG5SRRz0NcMgB69+ScDqIb2NYlU+i+B9l9f2/5SAHr603XAzHoAzQBhaT3Mz6GiAVRFe1rGVHwsvY/ul9tfDEBbTC4zC18KUGJREl/U42df1QCqo/0spC+/GrGL7s+w3ztAP/vh9ffbNy1AhyWf/TpQ54OnzCQWAzR/GRO2VP0SlfS7rlbufgCaPRBkxvXcPXR/jv3eAXp/nPk30+sydyKVAzSXoISl5peosOM1tW73Q9DsgSAG0GL73QP0z29d/4c/nj775+sGk/fHo18eb3/3PnjKTaLPzwygzATvBgBa3IOKWrX7EWn2QBADaLH97gF6+lP3NKYvfr/58In7AKZPqj2NKeRn9t1FmY6bAeiMg+96WrP7MWr2QBADaLH9/gF6+uQ/3uNzeOrnJz+6R+Z3/gg+OMpPYshPGVBmn/urTyIZQPdBEJtEKrW/AIDmqzCJjbKIMv8ItG9AbvVqunKl1QgD6HzZMqZCewMoUGESG60D0KDaRgCqRVADaAXZQvoyewMoUGESG3k0GaFC0KUUPWGtk3DiagHIbYGgNomkaa/tr2xvAAUqTGIjFyYeSBFdCskTVRPei78I5DZAUFvGpGmv7a9sbwAFKkxiq5Cf5FOQ/cI5inglC7MQ5C4NoHtcSD/HXttf2d4AClSYxEYxP1nAJMiK/cJwMn4VUU5S3H3YXkboerrsO1m17bX9le0NoECFSTx6R3kQoJCgcUQeSOsBVFTeAKrrr2yv7a9sbwAFKkyif5aMAZq5PpQnaFRcFlTWq9iHUL8MVo2fBlBVe21/ZXsDKFBhEuMp+FCytSJp4Pq7yNLelkyIh92h1XRfk58GUFV7bX9lewMoUGESA+KM7z1+CgiaBmhw5k+UDjYtCVDVZzIZQDXttf2V7Q2gQGVJjLA3vvX4GRM0hI8AoHGAuHS4bVGArq8pkwZQTXttf2V7AyhQURIB9nyUUk9siBh1BUKxgoWjjfsCqJNJA6imvba/sr0BFKgkiSnokQBl2SejHfxRpjhCJj8zJpHWl5tKA6imvba/sr0BFKggiWnmEQAFtbKPQE/xUSwT2P2UCCxqgcYQ8nJpANW01/ZXtjeAAhUkUYA8KUB9fsoACmiIIsQ2iciSBhhAdf2V7bX9le0NoEAFSZQArxnv6QPFkJ+SY0C4lW9QBp9T7ucM0PkpMILo+ivbG0CBCpIoOmLs+MlfqryCSjQAb5bwswJBzxmgFVJgBNH1V7Y3gAKVJFHEI1SIPwAVhaUazFQ8c4BWmkSqkQMjiK6/sr0BFKgoiRn8RASNywQEZaIXfInOAqBc8wqXMXkhqyTBCKLrr2xvAAUqS2J6IBIDlhjU8gPRnQKUb1/RQnrqH6vyRhpBlP2V7Q2gQIVJTEo0YHmCwrolXyIqWv5POyw1hMRwk/sHIQ2gFey1/ZXtDaBAhUlMSjZg8wla9CWi+ZlJ0IWGkJxuYv8wpAG0gr22v7K9ARSoMIlJCQdsNkHLvkQ0P/MIer4AtUmkCvba/sr2BlCgwiSOosaklAm5BGUaLMRDX4y6XZ/VGQPUljHNt9f2V7Y3gAIVJnEQOSoLAIppGjYgESnZg6HYxQHUFtLPttf2V7Y3gAIVJrEXM+6lSPDLFQNUiqCxmBCgfsicfOXwSpqs8kmkKjKC6Por2xtAgQqT2Ik9csojaBgwE6B0BaqcDKBByIx85eFLWrp4GVMVGUF0/ZXtDaBAhUnsxFMrh6BRQKoyfy98BkBFk0hhTHm+pJ13ywuKFS+kryIjiK6/sr0BFCg/iRN0AmqNA7Z7I0SaUwMANG4AVT0boIJlTFHQ2osQsqU7howguv7K9gZQoOwkTtgJqBW+yWBIUJWrWRGg6YX0BtDAXXsIK9tr+yvbG0CBcpPYPWWpAU8IuwEYYFNCEYWnQHEDuAACsyyuVQNo/j1PhAygmvba/sr2BlCgzCQOT6kbXkVKxPQKhiCO/op8hIyjXYFqAbTgnidCBlBNe21/ZXsDKFBmEkduZgA0FdMv6fNTOAsv5+e85UVlk0gl9zwRMoBq2mv7K9sbQIEykyjHpnsMGTLL/wxZS/GXXcaU2RmBgrhFy5iKluwTMoBq2mv7K9sbQIEyk5gF0AGUGI4oJjIK/opMwzL7IpIft2ghvQG0mr+yvba/sr0BFCgzidHxZYzNCHzhBrLAHICuo6IhZACt5q9sr+2vbG8ABcpMYnh8CQ47+9+Tg1VQCAlAp31nSBADaDV/ZXttf2V7AyhQbhJdrgGSDrhgjiWdz2EA1ynkZ7/3HAlik0i1/JXttf2V7Q2gQNlJRKfWzoYYFjRA/SPRFD+79wVfojD0DBUOIVvGVMkfbay2xjZtv8Xur2hvAAXKTiI632YnTNIAvYohh/jZfsr/EsVwLlfpELKF9HX8wbZ6/zil7TfY/TXtDaBA2UmEAB2FrveF5RFAA3k75gGUa222tIeQATRUxcsjafvtdX9VewMoUGYSWewdiQmTsHj/mY7k72EAmiYj31yJ3LpVhtCc8W4A9VVzgi5tv7nur2tvAAXKyyF/3HikvtBh6e7zfIAKyDgboF7lGkNo1ng3gPoygK5obwAFykqhz09IJfR9pr7fJNo86vkIPAnqu9vmAtSvXWEI5Q54v+kGUF8G0BXtDaBAORmM+UkSlN/iB+SMIgKecCky6kyAhu6zv8O5Iz5ouwHUlwF0RXsDKFBOBgE/CYKGnzmCck4xANMADTbWOQDtA6wO0LD1BtBANom0nr0BFCgng8yBIa0UMXAI0iYJ0Ggr1UoRVb1/M9YHKNd9BW2RIOvxc5PdX9PeAAqUk0EXJrUAyuPtiiOIEKAJRPMygLraJEFW4+c2u7+ivQEUKCuFV0juTvBV5olBUjigNCSIt2t4IyS7EP8BwteeRDKA+v7K9tr+yvYGUKC8HDIA7T7EYGABStKOtKGWMY1vZQAVYjYsufYyJgOo769sr+2vbG8ABcrLYYqfHEH5eCmf8a8YlfMqHIXHlnKA+nfir72Qnu/+2jKC6Por2xtAgfJy6BIN8nMgqAuJ5BXQiGM0P9svURzOKygBY9APQdnOffXvMOi+nowguv7K9gZQoLwcekhzR3YAUJ+Z/BS8BKDOXxEC2S+Z5mfkkCzcu6//HY66rygjiK6/sr0BFCgziRRxfICCs3YISep8m8HbCV4SEJJQZsFIewjljKEFZqe1u69tr+2vbG8ABcpNogSgYN4oqjRsSESL9p7gpFQ+QMsIqj2EMsbQEusjtbuvba/tr2xvAAXKTWLqENQ9AJ3Gb1Rp2gDhNe6O9mKAltxvNMbfJUCzVkuJ3bWHsLK9tr+yvQEUKDOJ9CHbxM8YoFGlBLeY3QRAS+83qgDQTG7Pk/Tvxa++LXbXHsLK9tr+yvYGUKDMJDLnvCM/VQBaeL/RfIBmH/nOkgFU017bX9neAAqUmUQGoI6oM/hZAO0+4kmkVHPnlHAV5yuv/mwZQDXttf2V7Q2gQJlJlAE0nMGoAdD+M17GlGxtqowk2hEMocwj2NkygGraa/sr2xtAgXKTKAQGMUsefqbCUPy8usIL6QmJ8JZDv7MBqE0iLWGv7a9sbwAFyk5iGS+CSknuUPzMM66Ot/MBqC1jWsBe21/Z3gAKlJ/EMlzk8jDYZwDtGyAvWp+fRhBlf2V7AyhQYRLnKhc7haCqTbezmURayF17CCvba/sr2xtAgQqT6KoYaxR5wGaifNIaeczg3dksYxpUt21GEF1/ZXsDKFBhEh1lE+TKlyzgUPqULBnXm9leR2ezkL5XZbobQXT9le0NoEAlSfQGZd6R4RVQUI0Aa7/xBIJl9aCo0iDtIZQ5hmb1FblrD2Fle21/ZXsDKFBBEr1BCY8kyWGL+IkWN1EE9b5EZElORZVGaQ+hvDE0r6/IXXsIK9tr+yvbG0CB8pPoD0rqQBIOW8zPmMXMqDeAymUArWyv7a9sbwAFyk5iMCrBICXHLcHPCMUGULoBOYUNoJXttf2V7Q2gQNlJNIBqygCqaa/tr2xvAAXKTiLFvakMyUCWn/kAtUmklCrzU7372vba/sr2BlCg7CQS4HPKkBBk+VkA0LJVOnOYoj2EbBmTqr22v7K9ARQoP4kR38IxSlPQ3R4XkPCT+lnjHM1givYQsoX0qvba/sr2BlCggiQmDms4DhJV+40Cfs78Es3iSXNz+VkAdIG74Ht37SGsbK/tr2xvAAUqSSKPIfZAkuRnSFCyAZkNxkYlWuLxRtmSdH+5hhpBdP2V7Q2gQIVJDORySXQmHlbuSgpqzfkSMcHTLV3gAZsFNBd0f5Engfbu2kNY2V7bX9neAApUmERfHpdyAXoFxDQgs8HQiNjFVV7gEe8lx8Pp7i/zLPreXXsIK9tr+yvbG0CBCpPoKeBS4QGoIkAFja3PJXmGHBlANe21/ZXtDaBAhUl0FYEJk9D5SPxg0voA7T5IfKtzKeffmEkGUE17bX9lewMoUGESXcUoaN8DfvYbqJ/sXB2g/ScDqNRdewgr22v7K9sbQIEKk+hKhAKnSDjAp12CODUnkXLAXX1uZimAbm8SqV5jNkIQNX9lewMoUGESXUlQ4JSJD5GmymmiVFzGFPNTMI00xx60ZQGAbm0ZU8XmbIQgav7K9gZQoMIkeso4AIUAdc72k0CpuJA+j5/VF9KX8PMMF9LXPCDeCEHU/JXtDaBAhUmcJCMQD9CcBiT2ZyDJaZMUZVWHUAE/z+9H5apekt0IQdT8le0NoECFSRwlPIZbC6A5UHJbLaxVdwjl89MAqikDaP96FjoHgEbHcDSLHMLOGlF8gx0XAZ0Q9flq2kPIAKop7b++tr0BFKgwia2uaMU1ne1zBhTb4NyTcsxP7kZSA2imDKAV/ZXtDaBAhUlsxPCTImj/dsZ4EgKUbgbVJi8A6W4AzZVNItXzV7Y3gAIVJvGY4Gc+u6QNSMXMawVVn3I3gGbLljFV81e2N4ACFSbxmAYoz64CvHUNSAc1gC7mbgvpVf2V7Q2gQIVJDGElvyFzBt/aBrB7DaDLumsPYWV7bX9lewMoUGESYw4KCToLcEfpMiYD6DLu2kNY2V7bX9neAApUmERAQRFA5x0hHsUL6QvDp6ppDyEDqKq9tr+yvQEUqCyJGIFpOkoYm2iAsFxZdFvGxLtrD2Fle21/ZXsDKFBZEmX81ARo6Sw/X017CBlAVe21/ZXtDaBAZUks5Of8M/gLJ8iFd1/bXttf2d4AClSWREBA0cHlbH5eOEEuvPva9tr+yvYGUKDCJAoOQGH9ufy8cIJcePe17bX9le0NoECFSXTXC7kbkmxMw5Mna9TgQhCXSXsIGUBV7bX9le0NoECFSXR+9wiuYqIjSPjJzYPnFa8s7SFkAFW11/ZXtjeAAhUmsZOHyxnn5VRI1IC84pWlPYQMoKr22v7K9gZQoOwkOrgKDjhr8pO+FyiveGVpD6ELBKh7J/1GCKLmr2xvAAXKTaKLqxBfFUBmAE00QNd9fXvvWU4bIYiav7K9ARQoM4nwpL0ivioDtDZaFxlCOU8rujSA+k8T3QhB1PyV7Q2gQHk59IG1eYBWPzhdYghlPS/zwgAaPM9+IwRR81e2N4AC5eUQXvWsiqhUyJxJpPrNW2AI5T2x3QCqKANo/3oWOgeA8od4ZexKMC9jGdMCB8ggXzPjZ/5mEPp71b5OwbgbQFX9le0NoEB5OYyYlOBnKUHpvRkL6VcB6FyD+QCtfhLAuBtAVf2V7Q2gQJlJjJDUfkAjeIHT+0YZDU4BtKB1Ub5m93I2QGvn+fG9SHebRFL1V7Y3gALlJhHwE47gBQ7/ugbIiyaaUNK6MF/zezkXoLXz/PgxQ1BbxqTrr2xvAAXKTiLFz2AEbwCgC8ww1Qfo3Emkynl+/JgjqC2k1/VXtjeAAhUmsdOVL7xr1h8tbkBOYQE/89q3AEBnLmOqm+fHj1mC1iDInN/o3AhB1PyV7Q2gQIVJ7LR1gDIE3Q5A5y2kPzeAzvqV+I0QRM1f2d4AClSYxE4MQDcwicS2oQ5A83pZIR0LTyItDtC8CxaR/1z7eTKA9q9noXMH6ELLa0qugfI/LpLlPmcZU42ELLyMaWmAZk6ZRf4z7WfKAM3dJGsAACAASURBVNq/noXOAaDMJNJxoQXetQBaZxLpmNHLKoeKSy+kX3gSyQA6y1/Z3gBaXw4/gz3NKKlslLuTbJuzVxS/VKEB3ZjNqOPn8K529AmgtSObTL7O4giUXkg/a7Ig1njwRh2CFZymuzuEB4d5+fKCll0ziBowp7JIw/EnOhK1I1Bdf2V7OwIFKkxiSsE4KWPHVGOqT14E9IL3H8W2zoE0WzgrX37AcwFoL3gub5NIuv7K9gZQoMIkJhQcaZTBY6rh1KenoaP1/WX8ZIvn5CsIeF4AxbNJtoxJ11/Z3gAKVJjEhHyASo/vfDk1MgHq81BgKa4wA6DLTSItocUAagvpZ/gr2xtAgbKTKGKAB9BcnHVya7ABoj1XoTK8EhXmAHSpZUyLaDmAzpG2vba/sr0BFCg3iTIKLAlQ8qklWwfoMgvpl5EBFNlr+yvbG0CBMpMoZRJ1Bl8GUCYE2JwPUPE5/IxJpDo690mkedK21/ZXtjeAAuXlEDEGX9SCU0jFACVjoI0MQMkGLAHQJe7FWm8MLbKMaaa07bX9le0NoEB5OQSQoaZVwSKmPJz4NZIAxa309zFNkLUxM1/V+bnmGAI3JBlBdP2V7Q2gQHk5jDEjWtgnolO0UwJLHLffEu4UNYFDXn0g5uqk2gojiK6/sr0BFCgvh9FxnfDWEgFBwU7ywJLfNtX095ZAPNXEtXVSbYURRNdf2d4ACpSZxEKAenXzdzJlREHFAC1v/0o6qbbCCKLrr2xvAAXKTaIH0PZFClCWX3gnpiWqKbEUApQqMJe/VXRKtoL7UbjZ7tpDWNle21/Z3gAKlJ1EZwT3r4sBNE1UahuMKiNgJuNXVhKg7I/CzXbXHsLK9tr+yvYGUKD8JIb8bAkq4Eo2QGHx/GnwKYScn6jMWQCUf57nbHftIaxsr+2vbG8ABSpIYsjPe4IKwHLFDn2wE5cvmvXCn/hW5uxaTwmAJp4oP9tdewgr22v7K9sbQIFKktgN36tYXM1EqXhnFYAe45jJ4lQrN8DP1CSSAXRRe21/ZXsDKFBhEo8QoBxbUoWifXUAyitqDtvKtfmJlrLzrTCALmqv7a9sbwAFKkzisTpAiSn3RQEah+dbuT4/o5sp+VYYQBe11/ZXtjeAAhUmEeLzCv64xzGogfdIK9T8EgUGYwfoTqw5hPDjPEoqVZMRRNdf2d4AClSYxIiaguNQjp/iU+aKX6KgwV5niCorDiHigXKyagu1yQii669sbwAFKktigMsMguKtDEHDBmQ2mJHfXu+fA6rK9gFqC+kXtNf2V7Y3gAIVJTFipZCgSJmVZn6JXB/i3wDWffsAXVJGEF1/ZXsDKFBREmNUOme/WwZo1GYDaI6MILr+yvYGUKCiJBLHmnMBKqkcQjtLgPrDx80BtGwSaVkZQXT9ZcXm/Gwfa28ABSpK4jIAldR2CmQ7xe2Lom0JoOQyJj2dCUEWs9f2F5Wa9cPRrL0BFKgsifX4mUdQp8AsVrtb4tC01h1CeCG9os6DIMvZa/tLCskfL5ltbwAFKkzifH5OBeUEdQqU0Jp3kETTHkIGUFV7bX9BmZwH9ObaG0CBCpN4jGmThzS3ZEDEVQAqX8/vSnsIGUBV7bX9BWUMoIM2DtBYWUiLi64AUM4jHakpoT2EDKCq9tr+gjIG0EG7BigouwZAaZN0qK6EAVTVX9le219QxgA66OwAmsM0VDTNz7mTSEfyMmg6WJlddRlANe21/SWFbBKp1/kBVEZQ6vgRkA1EcgoUAM07ACUePYq/fBn/OCwqA6hc1TFyHt23ZUyddgrQq0Bgu78J1I/fykRYew0nvn4G0Nb9LAjSqz5IzqT7tpC+1T4BmuInKBo0ILPBtDfcQ50AGUBb9/MgSKsFTmXPqftL2BtAgQqTOCk4JOQpk+InPKX3jMq/RDQ/AUDDcXfJAJ3ScUYEWWIy5Yy6v4i9ARSoMImjIsaxkHE4xNMy3tS/rwVQvI8edtvgp8YYcvJxRgQxgNa3N4ACFSZxUIAkHjJpgNEAHT6M9+JnNjxxENnvMIBGchNyRgQxgNa3N4ACFSaxV3RQJ+VnLkC9T2UoYwHaN5wcdnzl9bT6GPIyckYEMYDWtzeAAhUmsZfHRMGdPGRRsN3b5NYsRZmknk0iBTpXgNokUn17AyhQYRJ7+QAV3MlDFgTb3U1O1XKWSWrZMiZfZwvQy13GtJi9ARSoMIm9CgFK7ac3VQGo6NopHnUG0LMD6KUupF/O3gAKVJjEQSUAzbT0q5+WYxkbUp+frTu3imwZnekk0iL22v7K9gZQoMIkjloLoNMypqUAmog58FtL2H8Fqp/nMqZF7LX9le0NoECFSZwk5Of8Y7iu7mmpg8Fk1JZfat9h3LxVjounU2EjiK6/sr0BFKgwib6mUcyN5ipj/VQrUCD2nwH9QzDcvKUOxikZQXT9le0NoECFSQzk8pMnaKZf3IBagQJxLNrARUADaOuvbK/tr2xvAAUqTCLWCsM50eBid4ZFW5iGrgDQCpPSRhBdf2V7AyhQYRKh1jgg4htc7k633V3Ic8YArbEs0gii669sbwAFKkwilDpA59gnD0BVATp7EqnKjTlGEF1/ZXsDKFBhEltlHxBVYOtyAKdqbgOgM5cx1bk13Aii669sbwAFKkxio2jwpgBW4+h0wSNgouJGADpvIb0BtIa9tr+yvQEUqDCJR/YJSrgutTeLeK1/D4I5lwQzpDKJRPSiyP/x48cG0Br22v7K9gZQoMIkEve2S/iJr+OJG3Ac50Oyj4BLpbCMiepGif9jA2gle21/ZXsDKFBhEql7OBl6EXjLZN5pPCKMKy4F0PUX0pP9KPB//Ngl6KxmGUF0/ZXtDaBAhUnMexRoUCG9lWlAyE/6IfbTRlloifs632E6Kfn+jx+7BJ3XLiOIrr+yvQEUqDCJ+QAlCi8N0LpHpGcM0Me2kH62vba/sr0BFKgwiQmAxpuosvkAbcqGAHVeED+rEfScAfp4fruMILr+yvYGUKDCJPKPUuYONEmwChvQFfYB6oF0yauiBlBVadtr+yvbG0CBCpMo4KdwgjyTcENxdxLJCSG2LdP5TiLV4KcRRNlf2d4AClSWRBE/hYeCRfy8J+hYMRn7/ABafRlTFX4aQZT9le0NoEBlSeT4mQvQrHlyL07Izx0BtPZC+ir8NIIo+yvbG0CBypLIggnurASyODZD8nq2g7SHkPIY0u6+tr22v7K9ARSoLIl5AG3f1gEZjJ1uTL4J3qM9hAygqvba/sr2BlCgwiSyR3aQcZVWtOcdgB5LFtIzwbSHkAFU1V7bX9neAAqUncQeLgRluo3wGHHO386L7wXzjkCreHCxsvNVreNDA2oGy3fXHsLK9tr+yvYGUKDcJI5wIfk5ENTbVAkkcSgfoBUPcfEcTma+Kv7T0TegYqwCd+0hrGyv7a9sbwAFykwiDyq0dwm44fh1TGoCtOaBcd+AeqFK3LWHsLK9tr+yvQEUKC+HPKjg3hTbMgCDQ2GCFt/5XRGgNf/tGBpQLVKRu/YQVrbX9le2N4AC5eWwPkBzAEOEQgAtf/aQAZRx1x7Cyvba/sr2BlCgvByiYz20F22FJQSEmXbLrec8/ZJrkgFU11/ZXttf2d4ACpSXQ3Swh/aCzWF1rgauLQfovOevMw0ygOr6K9tr+yvbG0CBMpMoIyjYjCunEQN5myo38wcs6PbYJJKuv7K9tr+yvQEUKDeJEoISBLq6iionARoUYMq6uzyA1iRY/WVMma0zgGraa/sr2xtAgbKTyAKUvWuzAkA53ji7XIBWPQasvpA+t3UGUE17bX9lewMoUEkSE9jjD0/9ygl+xlayBof8FDBKVKj2EBK3bmxAXf9MnSlBKvyYSWd/nt2vZm8ABSpJIg9Qci8CaOoQrBCgx5CfSUbJQFZ5CIlbNzWgqn+uzpMgNX5Or7M/y+7XszeAApUksRCgeAYqgY8wlrTBzgVQCaKEIDOA6vqXVKryg86d/Tl2v6K9ARSoJInxuA/fE1QA/GQICktXXfmfW8wAqmpf1PuZKzI8+zPsfk17AyhQSRLhUaTo6SEEP9mFSf5uA6iizpEgBtBq9gZQoKIkQn7m3F00iSyM+cl8iYKC/ccxhLs7MlQC6JKTSNV+x8N11x7CBXUMoNXsDaBAZUnkDzkL+JmYcXIawEeKPzr89G8JFTUjUPUhlMlP+Riq90tyrrv2EC6oYwCtZm8ABSpMYng854NADoVqAIUMHwkKb8hnqlOqP4Ty+CkeQxV/y9h11x7CJZVsEqmWvQEUqDCJrqSHb0hE3atATgNEccKa7mds2W1JDTPtISQdQ48fL0JQ7e7bMiZVewMoUGESXdUHaMjPFQB67PiZGGjaQ8gAWiJbSF/H3gAKVJhET+X85M+nEekWAyhxqhfg+4S3ryYDqKa9tr+yvQEUqDCJvsr5yc/oANAtBlA82RAUHLs/p8dzZADVtNf2V7Y3gAIVJjHQHJpEdTl+FkwiRZ/5A1APoGFRdw5NhaA2iaRpr+2vbG8ABSpM4pHjx0y0MPjMXsYUHJPCop0adDabXYBGrfBWcaW6WevSmytbxqRpr+2vbG8ABSpMIsePuQdnHD8zF9I7McLj0bB6z897gqKWDO4ZAK02+evKFtJr2mv7K9sbQIHKksgBZP7pLRchq8FZTbmNS88BaL3lh67sVk5Ne21/ZXsDKFBREsMjxPDoDsKF4w0MgBvAVQW75AQFpWcAtOINMK4MoJr22v7K9gZQoKIkBgD1WELBhQOOuy/BprDB6bAzADpjEskAuoS/sr22v7K9ARSoKIkTa9xnbnYlCGpxyIlhTKMpaDBRvNsUNoVHHmx4sEG+jOmSAbrE7Fnnv0hUub22v7K9ARSoKIkOQG9TSzCPzNZ4H1euawBZNdqMW0Z2jW542P14e6wLBugis2ed/xJBM+y1/ZXtDaBAZUmc+Hl7G4KHQdpKAB23AX4KCJruvkQXO4m0TMc7/wVi5thr+yvbG0CBCpM48TMCKDzSmw3QYZMAoM5GwM8kQSXdl0h5GdMSknR/oUPvzr9+yCx7bX9lewMoUGESx2dvAICi09u5AB235QHUaYoAoOk7AHLy1R2ey8uLZABVlAG0fz0LbRmgEz1dgrIVuULuPo6fzVbBJBJGpQSgSWXkq4IbaoCgzBJL6Ht3A6iqv7K9ARSoJIkdGaaBIiIFV8jdx/HzfrtgGdMmAFrFDjQgXWSRmzh7dwOoqr+yvQEUqCCJAxmmcSLiBFfI3QfKMQDliku25ukMALrMY0R6d5tEUvVXtjeAAuUncULDYstVQnEApcvLtmY3RF5SA6ALPciud7dlTKr+yvYGUKD8JPoArf5nQsoFKAG6KvwUhrhggNpC+qX8le0NoED5ScxHw3yMTI6aX6KsftOF56TjLAC6mLTttf2V7Q2gQPlJzAZojQOxMYbilyiz4xw/i9NhANW01/ZXtjeAAhUksYifNQjaNmBemLlNyOkKx8/SdJzBJNKC0rbX9le2N4AClSSxiJ+VLgbGDa5+nZFSlZ7MDLL9ZUxLStte21/Z3gAKVJTE3AugSwK0YuyEzgSgygvpl5S2vba/sv1+APrbQb+bbVOYRF8cD0hipCCC9+OF9DUImo5Sw2sFgC4nI4iuv7L9PgD6/J3DpH/3T3NtCpPoiQUCRYwURYj9xK2c8wkqiVLByQA6x1/ZXttf2X4XAP309cPWAJogAsfP9IFr1ABcbDZAZWFWM6JkANW01/ZXtt8FQJ8cDi9+82eDfv77uTaFSXSUZBjHT/7u+H6/W2wmQKmSwjg1hpBrk01SA6imvba/sv0eAHr39uHlqjaFSXSUZg/YlwNQr9w8gJJFVwRo9Ii9nLoGUE17bX9l+z0A9NPXH/ygqk1hEh1x7CH5QB5hxvv96LMmkfiGrgVQQXPoBqx2+yx01x7Cyvba/sr2+wDo/MuengqT6IhhD82HK6cWKnUF1DVA7pHf0nUBKvT0tNyDOiQyguj6K9vvAaB3b2/uCJQ+kOL4QPAx2h1iZs5Ceo5YMpZpA3TBR8VJZATR9Ve23wNAT08OX6hqU5hETxEFuo8sIKgjzHBvdDU0s8HYU9IJJGWALvmwYomMILr+yva7AOj9Iei3a9oUJtEXxJ93mg6r5AI083CN8RR0AsoAqipte21/Zfs9APTu3TcOhwevfr3XNzawjCkSYKOIoHCnc3KdixvasziA8iSSAVTXXttf2X4PAPXX0UcL6f/tP15ff/E7v+w+fPajt66v4QdHhUlkhPgpO8KE+8EBbWabo7jl9SsPodzmGEB17bX9le33D9B/vm71xX9sPvzle+2HL/0q+uCqMImMSIBewSL4OMzb2L2pAdC5V1FrD6Hc5tgkkqq9tr+y/R4AyupP11/8T6fTJz/sOPnB9Vd+2Xz4yh/DD64Kk+hKcnTpkS/YCKkYb8wE6ExUYmkPIVvGpGqv7a9sv3eAfvbD6+83r/dHm/evf36rxehfvtccj3ofPBUm0VEINYeNt1eBwgITQWFY4CPl4uxjVSjtIWQL6VXttf2V7fcO0L98rz9D/+D6u6fTH67/pv3wh+iDp8IkToqp5pAuJGiwP49y+fysTlDtIWS3cqraa/sr2+8FoB//9NHh8ODRt8iHgbYA/eD6++2HPzXs9D54KkiiB6aAhCEab8MNUZnU+k5nXz4/a18v1R5CBtDKyjqg31/38+x3AtAn4xQSsaS+PVH/7If92fqf3/rKH70PXaG/6pXfsA5M/qde/qe20G244RQWugoiMm50MVhLXFzQT9Me1V1S1m6FaQlRAG34+cKrX3/jFZKg7fn6YgANyBTSMMRlzM+wjhsxQlYeBwGj+z3542Q+gU1b17CoQbsdpgVEAPTZw8NL77fvnr99gPfF/6ldxuQw80u/8j74pXMP4yc0dRsYfka7xyjexuitYxfX5RqMbLs9+ZPXoNVH/ZM4O4WvqdxltTvrfrb9Lk7hbw4vDXcf4WeD/umtL37/lD4CHZSbxIiHLD/DS6SDuo/T/0HNoPrYALptblHEzyyCGkCRu/YQrhvOAJpnvweAek9jevbwpehWzj/0y+hXAChD0LE8KhyEzKgOaqM406duT8kNPGcDUDYntd21h3DdcAbQPPs9ANR7Hih4OOg/Xw8rPReahQecyySg4NGfRHVUG8UZP/Z79gxQPie13bWHcN1wBtA8+/0D9LMPrr88XOMclnz260C/622clJlEzLkQgqBC/IGq7JWJ+UnSAiOvUdEt5DCY9hBa8FedRe7aQ7hyvMyvxd66n2u/B4DevX14c/zw9BCcwn/g3Kq5yJ1ICU6i4yHAT0xQRGH0nqJFZYDCYzvtIbTcrzrL3LWHcO2Aed+K3XU/034PAGUnkf7g3ur+2Q+vvzze/u598JSVQpKf4xkzGMvTJnK4O5W9+F7xJCzo3WXP4ACxtIeQAbSybCF9hv0uAPrs4eHFX7TvfvPVYBlT/8SlRs2Vzk/cBzB9UuNpTOAQ8ehwJjmOxQiMJKpNP5Wk2jM4tIeQAVTVXttf2X4XAO1uRHr06FF8K9Kfrj2Anj750f277/SHnN4HRzkZhAAd36cHshSBpQCl5/jzjjRoaQ8hA6iqvba/sv0+AHr68GF/J2eV3/bIySDNzyvZ8+Il/KSXRZWwojJftIeQTSKp2mv7K9vvBKCnu4/euD8CffW92T/n0Sgngww/ox1cBFF80qysxTm1aGkPIVvGpGqv7a9svxeAVlVWCmNMZgI0+dwljp8Fi8b3D1BbSL+ivba/sr0BFCgvhxGMcgGajM3xs/3rFcZMl+M39O7bA+iq7tpDWNle21/Z/rwBevdu8xuc9/93tfqvcoZgWQSg3HKpRQBK/LsQF9QeQgZQVXttf2X78wbop683PyGX+FXOfBUmcVQEPXdHVmgPdiT5Clf+5xUia2kPIQOoqr22v7K9ARSoMImTAuZFm8VaBKCSdkRe9HGr9hAygKraa/sr2583QBdSYRIdjaCJ+TljyRGJsMwGC46EDaBid+0hrGyv7a9sbwAFKkwi1IQcGkGcAMXiBuQEDJvF2RpABe7aQ1jZXttf2X4PAL1715k3evbGv197EomrAI4fCy6D4g9TA7ICBs3iChhABe7aQ1jZXttf2X4PAE09DzRbhUkEcqFTCFBJA3IrCBoSFSHraA8hA6iqvba/sv3uAPrs4YYA6jFzOwAVtSQqQNVYeQjFjTCAatpr+yvbnzlAgwn4VvFPeuSqMImxfFItxc9lALrRhfSg3QZQTXttf2X7Mwfo6WkM0DdBxTwVJjHWFdKMPxfVgNiVLS9ti6y1qw4h1HADqKa9tr+y/bkD9O6/ff3rbzx88Op4H9I3fzHfpjCJsdbhZ/Alkh1bChojbO+aQwi23ACqaa/tr2x/7gBtVGPeyFNhEoFW4af/JZKenYuPU5PuBlBVf2V7bX9l+z0A1FvGVEOFSUSiCIreFcv1l6E6g5/J1hlAdf2V7bX9le33ANDqKkwiFAbo9LbGcWk+QEvuRSLdDaCq/sr22v7K9rsB6N1ve330t9tZxtQKAHR6L4RUogHQja8z6wjU22qTSLr+yvba/sr2+wDo83e29TARTzFAo6PSmQQtPAItnmnyN9syJl1/ZXttf2X7coB+9E7zO0SfG39E46bCAsyUCID6q0Ff2hhA41/9WBKg9SaIEvwcd9hCel1/ZXttf2X7UoB+9MrAq+Fn3IQA/XDOreoEQJ/ct+LVZjHTGw+r/KpcYRJJhZzMByhfKHcZ07yZprCy9hAygKraa/sr2xcC9Il7yNf9kLAMoPOOU4lZ+LeboPf/f7NpWIXj4MIk0gqQkw3QRKnchfTzZpoMoL679hBWttf2V7YvA2jDzxe+9bv7d7/52v3R3w+abXoA/fT1tgVPWpLfaN+J5FCHhNREoDx+jldQgwZkNnjebfnnDdCh3YWdB+7aQ1jZXttf2b4IoM8eHg6fHzj4pL/7XBOg7bzR08PL4//nqTCJrRy0MJSatufx8+oKVTCAiuX/w1XFXXsIK9tr+yvbFwH05uBS6qY7BN0AQJvQn76u+jCR4EiRJWj4jtIVkteAzAbPXD0VVNYeQjnd57JY6q49hJXttf2V7UsA+unr/Vl7p2ef++b7pwGN9/u6S6L3R6bdeqLn/Wx9t226aNpuf/Da+23h5gLmR189HF74wYkTdQ20bU73IDvd54FGR4o1BioE6MyV5LOa5VdeaQjR7ZX78/8OlckIouuvbF8C0KcHdJpMAPTDYarpxX/yADpOQ32+KXwP0K8L1nASs/A3bcTuUqju80DPBqDzrgF6ldcZQkweDaCa9tr+yvYlAH0yTLwHFEMAffbw8NL9QebdT3roDqfw9zGag8+Pf9wR9B6gzVHt8/d4ZwKg9yb3we5jfKHOctTCJB4XAmjyHF57GroOiVhxiTSAatpr+yvbFwIUzHRjgA4n8ve72zc93+6R9/JQrDlubAAqmD2n7kS6aY9d74+MHzyEcM9UYRKPSwE0te5JGaC1UMSIzaQBVNNe21/ZfnmAPvhBXGriaoPOl9v/S868yXvhf92cuN/d1HkgfY1JpNtbfhIpU1MgFFL3S1SPRWmPmQC1SaT69tr+yvZLA7Q5KnztF2GpjprOlm4tfFLMw0T+tQn74aNH36pwP2lhEluNY9MhaIU/1BQIhFT9ElWlUZFJOUBrNM0IouuvbF84iSS+Bnr6cTtT1K26d0v5T/9wicroDB5nd+uMzppgmQLFIQ2g+XGqNdgIouuvbF8C0GZiyD3O+/U/NHSkljF99LV+Fr6dIFoEoDf9YqhaKkxiq9vbW+n4zBzDdNjcBo9RalBkFYDOm0Ty//Gp3FgjiK6/sn2FdaD3HxsoUgC912/efWX4tbexlM/LWQC9jzb/9k1XhUlsNfFTtj5e7MKEzWzwGKUKStYB6JxlTE5VA2h1e21/ZfuiO5GeeAtB+1VNIUBvvGmhZh3ThNnokudMgG7pN5GkAM0cydypZ16DxyiVWLIKP2cspPfaV7+xRhBdf2X7+ffCf3hwFyiNx5bdvLoDxg6oN8ON88Mx7PgopXKA9nci1VNhEvtBLgNoUIqoEcEuLNZ+OsWVJMbZR2O4rHAZ03KM5f9eMNE13WsO4dt75fpXtC+QAbR/zVJzG9GDZl7orr3C2R5zBseWTw/+MqZ+x7QOtD8Efdqe2s+7BvqkXaxfT4VJzFok45ciqgDY+aW6jydQSWCcC1CisGwhfXVwOQ0QGE/mtZtRkyC3t/kE3QhB1PyV7csA2s+tuzdjTis8G5rd/fRhB9Dmeum377c/71fKN0C9+11brJlVGs7s5wH04/soL4w/DT//JzoLk5jFJa8YUYni51VYAFVKG2cClCotylcmq7OUB9Dq7vWG8O1tAUE3QhA1f2X7QoCefv1wfCL933VbeoA29xS169l/0p3ZPxsLtsepT5t3DSvHe+GHWrMmkYJJ/ZkqSCKLJe8TKMdUIwFKed2OywDcJjrFSgFKFpccgS5Ksb0A9Pa2hKAbIYiav7J9KUBPd//9aw0ap4Xrw03odz995f548NvjLPzdTx81C0E/3y8E/fC+1stNwefvvDI+pensAcpSyfsMyvH1cGjKrBuAIS+8ckxLOZHlJddAc82ylDOJtIC7AVTVX9m+GKAq2upCeg51/vk5KEHijAyKfRoNI9DfTjco8nRFtQYUo4MkKtdQxjKmJdwNoKr+yvYGUKDcJFKku4LITOyf4vI1Ua1pCLrbo4JUdV/BHqowHySnUKkyFtIv4W4AVfVXtjeAAuUmkSVbinsMytiqqF4M0MiBCBv3KtpDFOViMNEqSvtpftVC2SRSgb+y/S4A+vFvXf1utk1uEnNAF3GLA5mzXRCqeQZUBNB4jopsCWXtbxIURId8UWUUq0y7AagtYyrwV7bfA0DVJ5GScGOoR03Bd0phz616tTRADBvSPwAAIABJREFUMfNAQRgx2IRdi7QfgNpC+nx/ZXsDKFB2EpN0I+SRjoYZMx8fB+wOYtxgQWhcm7Vkux8V4kKmypRoRwAt8Ve21/ZXtt8DQO/+5We9/vNXDw/+y8/XX0hPAo5XcKxIAosMEO8aTgKJMnS0sFP8Xq/7QRGqlrtNFlkoA6imvba/sv0eAOrq2UOVJ9KTiOMUnW0TWCEDoJ3DSSAsQ8eLe5Xa73TfL5DoBVemTAZQTXttf2X7vQH09GT930SiKJiWiKDuRrdEtMWv6n4kdkE74MuV8ocQXd7fGpShg0tE/r1w2Hlmsbv2EFa21/ZXtt8dQGscgublcKJg/oFoLkCPo81wRkxXRcBydnEXQENjthz8Tb2ouL8VfcrLutsAYjsOO9Msdtcewsr22v7K9rsDaI2Hg2al0Fm8V3AgmgvQaKtgIX74no3ri2lW9B1i+YmOOZm2CdXVIv5eOGy5GaHFCCKckd8IQdT8le13B9BnD9cFqHP7SNGpfCFAnY3Ba3QE6p/oJ+IG4toVfod42oabp/eSdnCtI8YQDltuRmkpgkjXhG6EIGr+yvZ7A+jdTYXfNc7JYABQlqDoFJ+o4VokQBC/iYrwEOa6x7Ur+A4lyoZbx/dscEnbdglQ8V1JGyGImr+y/R4Aevfu8CjQr7/xEP5iaKZyMugtXacnhTzCubq9AsTz2HJ0txxRWUwubp9bgu8fqu7Xkj5POmxb7MC3JGrZmOtEw9mt+b6BZF+X3BXyzr/LKf+csPVVCtCCWwawf40gM+z3AFB/If3Ky5j8W38igHpriDDSaIAyOKJCBZAivaYiqQ7G1YNA4gfyh02LqqWa4suZvOPbzW0tMfYl+rpk36O5d4BmJ4T0rxBjjv3eAPrCt+bzM+tvEpyD+wBtC0jUx0oXmXzTRQVRwmjRRrA/LEtNIjE5w+2kKyA5/3Jl9CnYWmbtSfJ1EZ+PRzV2CtD8hJD+80PMst8DQKsrI4ERoUKA0pgDtBEUoZ3joskYcaxoY/QhKkssY5LkDDkJNREm0Stuq6S1KQm+LnIaFlTZCEGyVJAQ0n92hHn2BQDNdaioswDo0RuVJOIg2ojSDAnYoqkQoFi00fkYFB7/wn6k5JUB3E6mApID0PKF9MnMCLQMQPc9iWQAzXGoqDMAaPizxiTjCLhR5eEYTxfl66Mo0cZUSSpfpKegSWk5Y7B8DNVoyUIAXXIZU6XLj529AbR7XQQvmZGTogB695uf/exnv61w9bNTRvdo3vXDktxFQY8LJnWHJYR9iDaSduNfmInK+5FNSmv3AF1uIX2ty4+dvQG0e10EL5mRk4IAff7O8MOf7U/VV1BO/0jeDeOSLwCHcGp/yhwXEXYh2kS5TX9hLmjuPrGmIThjDFVoyTKTSBn+uRXqNsYmkfrXZfCyPEDvvJ+oP/x1jcPQnP6xUOTWLjHUS+xOmRMlhF2IthBmzl+YjUk7kg2SaRyCc8bQ/JYss4wpwz+zfMWjv9beljF1r8vgZXGAdj9D/8KwiL7KMtC8HiawyO9OY+9qeOyS2JzYL+xCbI+93L8wG5J0ZBok0zAEZ42h2S1ZZiF9hn9m+W0A9JIX0uc6VFQM0Jt7Zr42nLg/f+f+k+T35RPK6Z8YkDHokuBrvmSi5UBUnHgr5oVfjIiGW1oGULIlJaoxhopa01Y6t3sZNwLQWtK2P3OAPgtu3XxyODz4wWybnP5JcQlAR4OvC5v4mpNxURFvAx0rCswF61UGUHZnniqMoaLWdJXOjSAG0Kr2Zw7Qm/CI82bte+FpIFLbJrCQ4Gv3cd/zlEVc7Mq9Hkv0pXejm+Rv7d+HT6THhfN20wKV5o+hoTVZUBkqaQ/h3Apyfkr+QAbQ/nUZvCwM0OYK6Jvelqc1roLm9I8nWWK1O4LeIO5IgYnqBUvsho7M0aO/cfgU/iYSLExnLU+o0uwxNLUmg6DFXais5ZYxiXpnAO1fl8HLwgD99PXwjP3+nH7lJ9KPwwiDioccwzQGoEzMIBy/Fxsyh4fuxqjfXGG6B1QJvp63sSpAxQQt7UJtLbaQXtY9A2j/uhBeFgdo8PjkeEuB8noYcCQEFQc5Dmk0QJmIUcCE+Vim9WvetIaCkUM6ZuQssxJVrS5ApQQt7EJ1LUUQYf8MoP3rUniZCzNPmwRoDyCSYjTDWPosDdCxVP/GfaKp4OqXpAuS2hmVDKBIBlBVewMoUFESaYjlnHM7Ep7Bs6uZBLbDG4efOSd4XA8E1XPqLAZQZxKp4BxeewgvFFf4dzWA9q9L4WUuzDydGUCnImmQkQSN/cJadDiJrSvmumu4UdIBTkQVPg72qraMKXN1T1dprwQR/l332n2pvQEUqCyJLFEE9IoDSxYxDZ/DKwnQNdUOGqDx1jGmaJzhThA94ytFJWotpM9eHdm2ZLcEkf1dd9t9ob0CQO+ah3+89n4J2bYLUAKJI2HKCDoqKBDUOFFFoQPTAhKg/gy9bwSWMaXHHdfNaKUpnYi4++XK5mfv7tqXdnyGliHI7e3wYPCUvTbBlO3XB2j/AxxFmAMAjaUBUIKIDKtgeUpRAX/DKSocFPOKMw2gADptd237d/FC+kRvkt0MEU30blCtMVTEz+h50msTdBGCTH/spL02wZTtawGU/PJF1W8OL71/ev520XLNrQI0C5OMCANQwNtANhiH97d5bxIHoOPPlfjdJ1wlueP6KY2jO4bAL5qs679ATPlcogF0eJ2HFzrfYe1nD1vAffp6yT3r0Z1I0w8aT/rGugvpx3PYGoLxQQF/gxCg0QZvG3cdMALoOEV1jIdQojeMroYWBDNgiWqg+yXuhUK/qbeSdedfPyQzlxjbaxNM2b4OQJl8h7Wf9DevPym5Z317P+nRdlEIR0kZJFDA3yADaPh5DDe+b/5PTsFHAB3e1AXo9EUSx4m7X2ZfJgOoqrTtqwCUS3hY+6a/ef1pyXPntglQARjbUUVtT428iHhSgIbVYmdvM6zfvQv56U731wPo0buGII4Tdb/Yv0QGUFVp268N0Lu3+1P3Zw8LLoJuEqAEF6WKVyVxBtGmNsephhENxcWC+t37gJ/uvwYVAep9k8RxqGuw64DMAKoqbXsDKFBeDhEVgej17smBBwp4G3CDo1rQnLb3t/pQOzIAnXMA6H2TpHG2AlCbRFKQtr0iQAtmy88ZoIS84zpqASRAgrvlRK+YDIMGIoLTm8ct0+7TKfzLg2jBXmqf/03i44zaDED3toxJZK9NMGX7tSeRLvcIlJAPUGIBJEKCs+mERm4UtBigcI38tPcU/+2jYFFcYmfwTzEXZ9J2ALqvhfQye22CKduvvYxphwDNJqhbJ7i0GMIJXr0MfadlAOPmeAIet9OPQ/bpGO13aqF/PePiUTWsjJPHUaWTSF4RQXnCXXsIK9tr+y8RVP4NXH8h/eXOwiOa3YaTM5B9XIRoD+mZaku6S6jb8PoNKh4GJpKZz8/SZUxeIVEN7L5HgmTYa/svEDPjO7j+rZzD+s89rgP1SUhoqNVViAkaH28SMXBw0jTVFrdHqba7tQBAcfEgMpXNbH4WLqSHucwz7tx3SJAce23/+iFzzoLWB2jVO5EWUmbCh1s5w2NJBkH9h1tAUNEB7RRDJpafzlu3dUwNV24PwupxaTZUoYrGkNeQOa3aIUGy7LX9q0fkpsRj+9UBevf24cVq98IvpMyM9wB1SJg6hhs+IYCK5AaRlXfe+20Rn8CHpXp5HQiqx6XZUIUygGraa/tXj7hxgJ6eV3wak6vfDvpdQWBfmRnvABociXEAmj4X8jM470weAR8ZRsSbJd6OwiNoHkdgF1FSPK9zmQAdRvhi9rJ0GED716XwEtd//s49P18reuIHBdA2pPbTmARHoF15d0sZP4PJn5NzQZUqTTEi3i40n7ofzoKRJb1G0xsS2+P8C8oQ0XsHvsUJdx2CjEN8KXthPgyg/etSeJkLM08EQP2n2qk9D9ThJz0XHu6bwc/xIKFdSB8czPY7p9IEItjWOUGQ/dD9oAeJKZlgD1WYD+I1IF2EaMYYX24Wu6sQZBrjC9lLM7I/gG58EmmWCIA+ORxe/ObPBv1c+3F2zbaIeQ7O4L4cjc7Nh/u/dNPg4HKqU8AtTHMKHR6DQiBI/ED+yQo5olbF5ajtKP84MmEYOKAPYjXpDm7lzA1RJucoaRmAidO/Q4BuexnTLGGA3r1dsqaUUWa+EV7CbTFeyhUYd39rH6BeAbc40/ioM3QXg78wXnkV+fvBCH+qXYzCm9GJPsLN+INQXb6Dh4nkBimSAXT0XyKomJ/7AGjZkihGeT3EfAu3ugWYfQJFxu1fG125YcfAsCcoxIwcYlebL9hCHInKhcgMKXocEtHuZKBMBQlfyoazNoBq2+8DoPMve3rK6iAFuGg7pEaBYuNmHAGAsoNg2kNFZXoa/IXD7rLlk/0SNN7X8PciqmREylKQ8aVsUt4GUFX7PQB0vL++lrI6SHKAJIXsMig/k38MAQoufXODwN3ll2AGDt4V/aoz70/3V2KGdIEAtUmk0V/Zfg8ALbstlFFWBzEHgreIF/7HaH8/MmnOTNvcq6CwZVyjiX1MX72P7hACu+NgOB01ljERwTiPOVIF6JkvY5JfZEz61wlTbL8LgN4fgn67pk1WByEI4veAGPSnK+e3gUjOeGWbDeG3khnSOKKzk+lsECR8onDKP04E5ce0w9NFAvSsF9JH/9aXywCaI+IU/t03DocHr2r9KicAnPcpIoazMxxz405nbJIj0ynLt4xtc15nQQw6X9gC5WGWLnASydFGCJKj+GrTDP8aQWbY7wGgwa/Dr76QnsRiYkXTtPMYlnWPbiLUjG8T/IxPw4Ka3t7csT9FoPMV9C6qWQc4pyAs1Yzq6v5CGsuYHGksQ3Xt8wkG5jtn+FeIMcfeAAqUm8SQBwEcOICGKLsaoRgfgvZ2QWV3CEeK+Rnagn0ySQB6HPlJEzTDklAwiUU1YwG1/NQmmOuuwG8DaPe6FF7mwszTVp/GRF7O7AowACU2+N8xNy46UwQNRmPJqwn5mTP+RAClQ9fj59FbRlUhXqa79hDuX5USYADtX5fCS1WynQVAweJwEqDxlg594AiUOfWNG8whyyUosU8i2BK2XLqJZfIX8teImOWuPYS7F60EGED716XwUpVsmwUocQg6lcH8pJkIJ5HEAIVjiRtiJcNvJkDryQB6PCuA2iRSnkNF0QD9+KePDocHj741/2GgpwoAhbM3415nJ8nEY5KfTvmwwXgwyQEqG4ZrApStvhRAZdHWBmjYqjMEqC1jynKoKBKgT8YppBpL6guSGGIt+CZTO2mAjuv8OID2FSoDVDgOVwQoX38hgArDrQzQqFXnCFBbSJ/jUFEUQBt+vvDq1994pQ5BC5JIYdBj0rhz+gBwGAiC06lDmE5bgSvtkih2RPQvmkQio+ZHWGYSSRpwXYDGrTq/SaS6/sr2uwDos4eHl95v3z1/+1DhvviCJBIgHD7HSEMshCIKc7W9rc7etM0xeSjj7ureJ/MlGNhskVSG3GVMiabIlTKd3NccwqBV57eMqa6/sv0uAHoz/UJdlWeDliQRsizEWAE/ZxI0XrvE2QR1mcY4dVL5SvUvWSSVIucklm9JjtJ/l8F9KwCtnAChDKD961J4mU0zV4KnMT17WPJzn76KkghY5m6YdiTxFw4FPwStsAZjABQ3D+xHOxP5ErQgUSQVYZExJE7cdgC6ttpmGED716XwMhdmngTPA63xcNCyJMa08j7HgCL5hnjIVI1DpAyAvHaCGnQ4A+h6Aq1SI0jXDgNo/7oUXubCzNPZAPQq/Dx+4RHbrvzZHhJgTP20B98H0G64H4RbHKCp/SVjyAlGxJXljeq+pGayZaJWaRFkaIk2wZr/1ZvUz7bfA0Dv3j68OX54etjAKfzVFfVTlpBtLhoJXIwbiABCD0peKVCci7Y8QBMFCoawE4yMm85a547shXULakb7lQAm+14tr6b7FZeVZtvvAaCbm0RiABZvHHaNW9BX09mS9OAbgBVHoPbHsRafRIraFygfISCdRKl0LNR9QZfTLWNKeP75JjUk+lqtoFPdG5uy7XcB0GcPDy/+on33m69uYRkTAzDMIaZCsD/lxO53w/ldAJ7E/rhUmK8oChdYWoRrYDZCnGCpjqcFvi7lQQtqXjxAq95an22/C4B2NyI9evSo0q1IBUlksOV90yCIuApBAWL0w/1UvCB4aIAUmQbdj4pGmxI5TBXhGmgAVdD8xNWRATRH5K2cHz7s7+Ss+9secjHYcgQKxptOzq44uvc+tmVb4oeCzY861nwxceNgcdD4olzS+Z0RGASrGhfEX63m6tpMQyeAardEqC0C9HT30Rv3R6Cvvjd7AqlRXg/FR6BX8ACRuCwKppBkDmGwW/cDWGvlxw/71n0zucb6h2B0JLFgdS6uHYGqqGun/jImOwKV6zweZzdPTcRoBLn7U/X9YA7+jkEzUfiga8NXc9wLmlsZoER9JuxSk0gyrT2JFPnnm1RS2059gF7iJNKnr5dNlW8ToCzPcjVGpDySfvEanWCPZxVFdzX94057VwYoGcDd6O9eaBmTULnLmBJu2c3RBpi2//ESlzHdFK41CgF6927zG5z3/3el+qucszVFJEywY0RY1DQKoGhbqxCg6PB3LYA65An2L7OQXqrMhfTJ5OQ2Rxtg2v7N/3awkJ78s4MAdzeHSgD99PXmJ+R0f1SOh1S+4GmcXwCdSHv7o1odAsMz+Dha0LcIoFe3V6GCPz0RKZFA2NNUxofPukM47+uSn56kf71QRfba/sr2lQBKfyvi+h999bAjgEZAqSjCBC7LTNQCAB2KI8NeMUABQYOaOFIqgbDRqYz3G84IoKm+lfhXi1Rmr+2vbF8HoMy3Iqr+5HD4/IeVALqQcjKIeVJHlAdcThr/DdwSt1fUyT/wmxTxsyMxZrhjm58/1OpUxvsNBlBFGUD71zl44b4WUfUnL753emoAlYi0iI9Jj+DQL6wk9XMV8vNqvNwkqJyVPm4TX+UU7i1rSpEy7eamC2gjBFHzV7ZfH6CN6gL07l1n3ujZG/9+1UmkJJVmiLQIN07l6KYJWop6F/BzAugxCJqTNNRGbhNfJZ7EKmtLgXLt5qYLaCMEUfNXtt8DQHUfZ5fmEr30CFWPz2qT8agG80aoNUSA6V24YISrnZc95MrXGT6f6F0LK9+ufgM3QhA1f2X73QH02cMNApREFrvzGB3l4XrwS+QV45uJj6OCtnSv0Yo7v7miZBEu0cZUysePp2CHuDVzRdix/rh5M5pcjyBFjTCA9q+z8MJ8bWGIagANJuBbrf08UAZNo26JyWtGU+jEJBL8Enk1+VYe4dCBZcCKZbe5wlSl66cC+vs3BtBEA9DOOW2uRpCyRhhA+9d5eKGTD0PUOwJ9GgP0TVAxT5lJpNk06DYboFzgYzh0T35DjgGW2je82ViW6VN4x0fUlOTgg+Vw1Zyl0dsCaEELZjW6FkEKG2EA7V9n4oVMPQxRD6B3/+3rX3/j4YNXx/uQvvmLosieMpPIsalVOBEjER3Xt2xzHLQj9+KnE4rrko81FF+ap2hzWDTr5rxNAbSgCfNaXYkgpY0wgPavS+EFhljwGmgN5SYxRaZwKZBERxag/vHCydsUFGZNQGQ6Fpzi97vP5klY7Jj7eIhNTSLJOzmniiMDqKr9HgDqLWOqoewkcpC6Gi4dJgqBOnhPZ+q+Z54GlTfHNXYptd8db0SJWLJSx2PuA8o2tYxJ3MlZVRwZQFXt9wDQ6spOYoJL0e08aTE1OlP3/YnG5KUBdM58dokiO3EnZ1VxZABVtd8XQH9bFDVSfhIFcMoCaHwLpf/9nj41/6MBDlHINaTvkvc53BvUQPWRRIVa9R0vAujqQj8JJenkvCqOf1Gtao0wgPavS+EFhqgO0Luffu6f2kVNr71fFNhXfhJpohTJf4hHPE8jjZNRFMcP3JC5/5lGXhCF0/APx1kCtOQqQkGVyb+sWq1GGED716XwUoFnkyiAPn3YPoOpWRX6YP4qpnmn8DnMOhIVPIAeUwijw7uFnVrssfAx5CdYFOUWDxpOZiloFit5yTb/smILKf66yFs+p8roX1ivUiMMoP3rUniZjzNH9M8ad8vn/+Wdhxo/a+yzgaNTRB5c2gHo0Q9ZsChp+OiGTzfMi+F8ogtf8SfdoGleM2FRWf5FpZaSEUTXX9l+FwC9Obw4nLnfvV14dcBVZhIDNHBwEoIrAGhZOK99Rx+g7GVQEGP6EJq4n9hpn7htYTtBUVH+DaCa9tr+yvZ7AOinrztHnavfCx+RIQE2ClvHIyBo1gWBKB7R0H6Cm6wpChIDNTVtTvU27ZHQzDEEfcTmFEHkAWaK7f3yrTCA9q9L4WUuzDxt/mlM0Qa5/KoL8HOINS4QoqqyUcKdfpPFmWIpmehGqHljCBrluGfe21xbXO9XaIUBtH9dCC/7PwKNwcVwjVEQDPEzioxmiLgx0+0aSYdDsXM4sYXbYHmieEomuhFo1hiCVjn+uU/XqS2m92u0wgDavy6El5Wugb4M35cqp38h00qWDrn84cqQsPYrjm1LL0V3KzmVgzi4vL9Fkqjb2I6oTccE2+eMIdgEul1Auc93rC2697NbIbmVwQDavy6Dl3UA+vRweO137buPf7z205gA5koImjqv7iKHPn35k9+QoWmCm3miVjvbk+XdDaJE0QCFBKUb4G8zgGLNbYXoZjADaP+6DF5WWgd6czgcHjx69Ohh8e99esrpH+IctYOTgKDuTj/HyN/5OaO+PeFw8OoQ4yzYF7rA5tCJunXLg0YnQ+Huu23NVJQ1ciOl3QJUdjutAbR/XQYvKwH07ifDw0Af/F0Fm5z+hZSL2CJVkqB+UKYN3Ub3XtDjwE93OHhViIHGdYpsDpOoW7d01OpkKOznPc0v3ZB0xJxu7RagwgcSGED712XwshJA7xH60Rv3R6Cv/kOVxzJldfAq1rCjv6UblACavqx4f2BGt6Df6t1Lf5xWRqFa1JEk14BcgDoEJcKnY2E//2l+gpbAkGijJMBeJ5EMoCL7vQC0qnL6dwU0bB8Wc4oUHIH6FWMzsgX+1jFMOB68CigsEVlSj00VlT5JMFyEuAQsE66XEWmvy5gMoCJ7AyhQTv+ugI4uBaUPYRq+rLBmbEa1wN84EXwDAEUXKZ0IgmC4yAIAzbieuteF9AZQkf1uAHr3214f/e2q60CvYjlbfXpy10V71B3dz0FNj5/T0Ais/W0jQZ3xEMQ68odh4T7vE6rGporMnwyBsMQSAEXCMNktQWwSSWK/D4A+f8f5Ubl1F9KHlAmAJNpxhR6gLDl2jVsQNWoIPAGUbRbXM287+iDJFL0ZNkISZCWAEjjZL0Ek/Nxx92X2uwCo/+PGL+kCFG2bBijeddV9W71NorP/KGbUqPGok+EnnEkhzI7BkWRQK50ouIMvAUtPmjWJJK5HHZDtmCACfu65+yL7XQD0yeHw4NXmtznfeHh48O35Njn9A5zhYEdO2kdbZddPQV3/83RpVcBPuo3TTvI7NNTEUE2GOZLHqKiY14Cs2oWu5CVBI4iuv7L9HgB693bzNND7/7/ZsPSl+SuZcvoHOCNiXUiUqDjmHQwaLqSPwxwHgobFAm+iS85O+jsU1KTzxCdTmHG3ATm1S10NoIS9tr+y/R4A2j9M5MnhC6fmpqR1b+UE5/AMfpwPCWo5/Ixnc4K6zG8yTWMeNyjwhgWj8uFfGKeCTlNWfql8Ow2YFU8oAyhhr+2vbL8PgLbzRt0PLZX+3JKrvB4yeAuh4X4M9gXFXX7Gy4mCsCeSes6Qxy2KmpXNz2kfWwPt4cKmsz1uMoBq2mv7K9vvCqDN2funr88/h8/rYYwNEj9UKRDG4Segjx/2hD2HuSmynVEXQDFUPK6KDJKHoHzgZLbHTeuMoQucRBLZa/sr2+8BoHdvt6fw3ZNAdR+ojPmJiyUI6lEi2uTtPYVlMJuYfVFkqiFUz1FNurjzWTLTi6OMm1YaQxe3jElmr+2vbL8HgJ5u2quf3aXQtR+oHEMDsgfCJVHGwQSuO+Y4LMIsWj/ifYQVwBXRc6qPsML0SbTWkA4ydX8F4ZYaQXT9le13AdBnD5vfg797u+HoTYVp+Jz+xcBA7CHYgpCSKglYFp7C5/2FqB7FH6iO4w5wVYYP1HmxtHmtdMeQEUTXX9l+FwBtngd6f9z59HB48PDQHo3OU07/YmQABlJkvCLWy4PjLI5M0SRS3p8Id8kz5fsdlx/6QFbq3pMzM7Lm9TKAatpr+yvb7wOgp183J+53N+2NSIrrQOMtEywQccjV7YCFDB7jWfi8vxHqk2eZ6veVW6P/P6oZ5qQIoBGZDaCa9tr+yvY7Aejp9K/33Lz78NGjb1V4ImhWB2NwxWiJadNDhLw7KECXWwrkWDSBU9yrdLdBOXbj0MUygIYygGraa/sr2+8GoDWV1cEYIhiXSPTdlf4BnFsO5Zhd5RkGK+sUXyIqhyuHeTKAVvBXttf2V7Y3gAJldTBiSAKarmh+egeNA2UomKF1oKiJhX2qBtBwY9EkUiwDqKa9tr+yvQJAP/ra4fDgtfdLyJYA6N2//OznJWFDZXUwg5dX4dl8xE/oEB+nBUWjZUwCmOV0KdxH9ZrzJApW4Cc9hsRdzshN7D7881UaoFid6UYIouavbL8+QH/c//rbDwrIRgL0+f/Z3IT01fu4L5bEDZTVQYKUhILiZQANy0YL6RMwy+1StCsfoGTB+fwkx5C4z1nJidxPMwOUqjfdCEHU/JXtawH08b0IB19P2wfOPX+76LnHFECftNFuKj1Peea98Jyi4j5AscHIT4pXToNRpKgGUYbsEQjv7DzBsv5nImDkmkg1FvH34pNaUhC7n9gA5YETGkw3QhA1f2X7SgB9/JgiaFC5e+hc+wzkgqcmEQB92mLz2cPDS78pEfsNAAAgAElEQVR//vrK60CPcwkKuBKI4OdU/uQU7aO6R3VhBeDnfyaNYI9O+CAaOvpl6A9Zwn+vZFqzCxLuJy7AjMC8Zra6mgyg/essvDx+TBI0qDzeqn5TAjryVs5m8efT9rLA05XvRGq6iLjSfbfBpgAnDhdJA4KfEKDDL8i59f3yIAD1md7uVBk8UWEcD7kmk0BrwwCdFZnVzFZXkwG0f52Dl8ePaYJSUSoCtH+YyE3/NKaVHybiD6GQFGibv8iSm14fhPk5rpnyAIrmtd2RFkVAgxEYhQWH18nS3xf7B9GIlOTl3u8+9hSEKLAekswBtLRXAi0YOksG0P51Dl4KANo/BDlT3OPsPn29fRLoGQDU+ziwh7cgARqMIWphpVMojHCEgzEyigv2LwCg0cCWBYuqpcVUEUcssx6zbADV9Ve21wLok6LnHnMAffawvaq6JYA2O+ON3id2IaQ3PGD4aBBNMKNaG7UQndP7RVHlyJFc0IojVQEoW0ccsMR5+rvFk0jT27JeyTRE3ghB1PyV7ZUA+rTmMqbuFP5JF1LjGigB0G4v3teLOmJ0avqf/PBggE4sI1obNTFuMtECv7BvyE5ywUgVAJqoJI6Xbewe6UfLmJwPRb2Sqg+8EYKo+Svbrz2J1OnpwwdFv1xETiK93Ey/N+TUmIU/4d8kGnYHW71PHED5OAH7plojyqjmkjGISRCi8riJ5GdEXq6DZBMYMQ2nPGcWHBQAFF0iiT/UVhd4IwRR81e2X3sZU6snZcef3DKmRm+e7t4pXKHvKz+JJDdaxWgZxAA0igcM3E39Dr8U35ywbd1u8opC3Ij29QSvgEqwAViTqsI0Zk2FAB2bE7Zp+eZthCBq/sr2qy+kPzX3IpVCjl5If6+X24mkskNbX/lJ9LEYFrgiVQ+gwx6vFDF84xHulGOuyYaN6N6cvDtSs7AWsSZZg27MqkIADf+209ZFm7IRgqj5K9uvfyvn3c3hxdJpHvpWzne/8d79y6f/U9k99oGyk4jHzjE6KInlX7F00RVF8zf4sHJQ5BSCTQoUlgiBHtYF4U+BR9ITRhOVZZu+prxJpKA16zZqIwRR81e2Xx+gc35zY4tPY4IL6fs98Q63TPv/W+cxdR66mGBR5OhcHpXBCgqEh8RxXRCenO5KZU5eErQkp3J9jUk6oScRrNiqjRBEzV/ZfnWAPpkzS75NgMZjRzIBP+yfTn6Dg79oIIY1R51iuAL7ROu7TwFA47rDrU7wAmzYbXnmxEkGdXXG0JAjCqBrtWMjBFHzV7ZfG6Cfvn4YVLAQlAPobwf9Lj9uoLweRkNnGD1oB1civB4qD4gPNokhHY5ub6cP0Kiuuz/eGfnKU5dd0tlwikqmotUUAdD1/Fdzwvba/sr2awP06WEJgD5/Zwpb4XFMeT2ERKN2BAPM2xRNKOVF5IMHW8MODO/PHqCycBUVAXRlhG+EIGr+yvYKD1SeIQKgzmHtNgBK7mBLxDPycQ0hQ2FZbxvVG3gGjwE77IvylfIoKMnkwRtDcutaCieR1nPu/dc2DOy1/ZXtdwHQJ4fDi9/82aCfr3wnkgxpAsCNbMKwzDJLl6W6E08huaW9I9RuF8gX71BSMmqJ0xPwONQ1T6H9ZUyr+Y7+qzv69tr+yvZ7AOjd20U31tPKTKIUatMaI1xx5CcN0AyrVFmyO+gigr/XO0KFQyhhEZSUlvPKTj3ZCEBXv/ra+69v6dlr+yvb7wGgZU92YpSbRCHRIoVlBn6SAJUYxcaBCdMgqnnu1uAqAxxCRKf9bUQBvjWodVsBqI5C++Cvs7j9xrq/tv0+AFrhZzxcZSdRxLFYUaGenxRA45jt/5POgYmQMFTbwxEa54uo6W8jC3DtgSYGUEfRv29L22+r+6vb7wGg/QOV66koiUmMASXKhhupeNM2Otr0Of385iCoqPu4W3ArFV7sGBTfxiSSlnz7aB5ycftNdX99+z0A9PSkwhOYXBUlkYDhOKLh17pHGl6GzW04ukdkJ6IOAlZ/7xPbIVA92X1cHwfFRaSOQVePqWVMkj7M0KYI4q9CW8V+S91XsN8FQO8PQb9d06YoiQhdjoivdbsL1T6CA06wt2uAAwnk7W93pvmxQO1093EAHBUXEToORaay/EJ6WSfKtSmCGEDXtt8DQO/efeNwePDq13t9Yzs/KudRK1oj776Z2BG/Q+f03rlq2ODI3d2YHF2oMlWudRdMIkURQfi0I9E6+PcKu812YY42RRAD6Nr2ewCov45+9YX0krnx8IsdgAQfN7KQ4QgCC7ktyeoNW+6IhxDf7GM5QFHX0N9r2McGTbpJtCmCGEDXtjeAAuX1EEAtGuTBF3vczlYCR0+wWPwlomOlBxeIzxUjhpBXE0UE4ROORALgGBp3cv2Q+KW1LYLYJNLK9nsAaHVldRAwLR7jPkAFVQByiJrH7kvkk4CIEwwutz3jK2VN9To9hHBEED7aL8l22/2gMJtDukkFmld7hkbjwmVMtShrAO1fF8KLAfRqFYCGICACRfyc7mkXYAf2Oheg3mZQ0qskyPYwh5bqPRsl0YNE10prz9BkXLaQvtpxqgG0fz0LnSdAm1LwDB6WJUN4Vf033AkxGWpoUrIg3+scgOZnlQk2FgifiE9OtlExslomauXScntfUr/emb4BtH89CzEAvRseB/rR3657DTQYiASKvLNnVGQaiShCyIbpDV+OiTQdFPPlUr1eBKBMlWEzShiqHocp6G61jtWRa1xCkIpzTQbQ/vUstPnngR7x8Gw2tqgKKrhHEtM45Mf3+MkLQCOECkQDVHRhzxnByXzFnZDWoC6Duq/cjL5XLg4OE1ynlYvKNTaAqtrvAqD+NPxLOsuYws8kv8CH4NRccJGOKHaVasJUlwGo55HstSBfuA/JClylaR/MEleZTEyu5tWeIdfYAKpqvwuAPmnW0b/xsPnvUOOepNwkBkOIpFdXDtEsLJQa30z45O7+XRKg0Dhog/sd4sRHwuVZf2cnzlKyLp/gOq1cSq6xAVTVfg8AvXu7+aG6+/+/OfM36wYVJrEVyS5u0MZ7+PGdiBhuc/Y6m0X8xGzw9oX5ws3NzCl0j3vo97YgeG5VQSvXkGNsk0iq9nsAaP880O6RIjcNRmeqMImNSLQh5VRyDROF4m3jO2/HMIiGvZ4Rdkb7gnxRzc1MKqgU9sdtdZ6FU7yocVwr15GT/qL6toypkv0+ANrOGz1tn0v/tMLT6QuTeCzkJwNEDAeqJNo7bQt3TYchQznXBztHBsdwCMW1yDi8oipOHJih3GuslM+8Vq6l0biQIJX4aQAdXs9CCYA2Z++fvj7/HL4wiXn4FM46haVTsY7U9qgufyGMihFfryWuYKQDQZFl3TipHKypRQiSgbeNEETNX9l+DwDtH6j87GHD0RqPpy9MImZbP7q5Ec9VDAqzO6MSQUP9SpUAelUVoHRhN04yCStqCYLknGBvhCBq/sr2ewDo6aa9+tldCu0wOk9lSfRHMyZdMOSHcS8oeow9QqRERYJ2BnUSU7FEENCGigClG+/HGd+CHK2sBQiSNcWzEYKo+Svb7wKgzx4eXnu/mYb/QgNTpVP4AE8E6oDiyrhMulhQJmhmWDa1lgUGgY2AO+kttKiocZzhHVdjHdUnSN4io40QRM1f2X4XAL2nZnP/0dPD4cHDQ4Wf9yhKYjiUWdTJCxMzO2Qk8tAxLpkaphSTsG2wk9tCiowaxxnecTVWkQFU11/Zfh8APf26OXG/u2lvRFJaB0qCjNK0BpMr6jl5m2FpkiYopmSUIjAJ2shv4cyoDlBxuBrLabIzgOr6K9vvBKCn07/ec/Puw0ePvjWfn7MB2m8PiBbIWcVOLMuh4NRvJiPLOSPjZ3oiPtqZ2sK6ET2g4nA1lpLjZwDV9Ve23w1Aa6osieRAjnDT/N+9D4gBIxEqjppgTyFm2DrDzorfYb4PiUr1mpHjaJNIuv7K9rsA6M1r71e1KUxihDCIum7HdJQRAPQ2Lu1Ixk/hcVoCOomAY4mqQ6iAn+6vOq8jLy21uu8S05Yxyf2V7fcA0E9fr3D7pqvCJIbEgbDri7oAdUocw1vTfSdnYy5A4+O0BHaSAccyVb/D+fz0ftV5FS0BUJ+ZtpBe7K9svw+Azl/66akwiQFyYpRigDp1j7csQHlopiAK+cmjMQFQYufKOFt/DC0A0Kyzdk8bIYiav7L9HgDa34lUT4VJZEg28hOcwk+Vj7csQDlWOgYsQUEw1CXYCboQ2CxP32ztAKB580aeNkIQNX9l+z0AtHmGXdWLoIVJpAgX7A0nkZxItyFBPR8y+okowTVaiMZ8gErM62r9MeT20QCq669svwuAfvyTw+GFV7/e6xtbe5gI3jlNIR2d6j5B3T1E+LYBXggRQYEJbzTtBWapjV70bNHVuj0lY6iwKW71msuYsgDqF9sIQdT8le33AFD/Fz1W/02k9NOY8M5xEZNX2yOot4fkGnkESiMCUTppFBZnQkDzRJso0dX6PQVjqLApfoDu3eoADcpthCBq/sr2BlCg3CQOowmB5yq1c4gyfHZP4/09cc2+DAlwr5nOBpfSQQm6lVEF6BNVifZE+YMb09VS61CZsHTMfNWdREozNCy1EYKo+Svb7wGg1ZWZxAgxiD8sm7y6IdzoikMhchILtnIYg1ERuol+5+LyMEhwnIdbdQxKEclFJeg9qbCJmnmquowpTdCo1EYIouavbG8ABcrLIYRMBCCeTj5A/eFNVhvfUAAlWjkMQqoIbiEIi3yCuTLUDTJ9XHZzASrZtyWASglqAA38le0NoEBZKfRGI80gTkSYyMAr6mwnAIpbeUtecaMbFzXQ2xbkaygUVMDtSsKM3g/3+DkiwqY8g1i8KhJEQtCwTNHMfUUZQPvXs1AI0E9fr3DFM1ZWCv3RiCGUUFDx6I/dqGg8Vw4KXpEAvc0DKHsK3nyM8uX0IWh03K7KAPUSSIZNeQaxeOkCVDrztJgMoP3rWWjjAO0GHMGhoJy3IaakO3aDXfTmKCoq3J6543FKtdffS32HmMQw9Yn2SnbHu6YNfFjWMqNMo5UBegT8LCJoJfAaQPvXs9C2AdoPOAJDzohObgmOfryi2JjYQrYSDzuyxc5e8jtEB+Pqky3GMQQGTgKJipKjy0SASWsD9Aj4WYDCWoeuBtD+9Sx0DgBNETSuw8wThR6kM95AN/IKj550M+jvEBnM28BZMvu4+PAqA1eR7IrY2lFNgshw6F8ALSPhjENXXwbQ/vUsdBYApXUE1wcFdSYXcmI7Ami6jTkdQUW97xAM6E0nyRJI7UlURBuYihIlvCfVJMhVJtWKATrj0DWQAbR/PQudPUD7V/9jqtJoQ15WrAFQSSO8os53CEf0Z7tkGSR2JOrBDbxpSgnvSRUJctVdWpFXMIAaQLO0RYCmr3mCd8PQTNX1B3C8FX9yP5MtTPQC2iHT++8OERIXdxtGJDDeIYzMF8+TNNb0dcnlUSKvEs09gzeAzrU/e4DGWvtWTognd+wnCiSFjdxt4X5/Hx+V6ES4ERXtvztUTFw8aDRMIF0VxT6FG4jSmRLGGr8uuUBK5VWkWVNIBtD59gZQoMwkYkBNg4HdLxD2cTaegt2CZgkOQP2tqGj/3SGDwuLMgSPsHBV3LOCNIbZ0pmSxhq9LLpGYLOQ0shCCNolUyf7sAfpgfIyd2uPspm9+85WkgTFHcZwpdvclQnv4eLgPw+72xdky1gqD5AIUFCfbR4d1pDuG+q+L6JiOzQKVxkTMwt7bMqY69mcPUP1roNM3v/tSMiNjhqI4Tuh8gNJ9GPbT9cIoFQDKXwZN5f9sAIr/PP7eqMOpoKW9r8NPA+jwehbaJkAHggwjyB0I7QuGUaaO4Jqk+zkcf1wgIHkrsiaRgC9uiaiRRP5zChPK9XTcxQAl/nR4/6Bk1I0QRM1f2d4ACpSdxPZ7Pw2haWx4LzN1jHBEF6R3zuOn6+t8h8jIyJdoiqSVRP5zCmPlm07uUoCiHPquoBXpsBshiJq/sr0BFCg7ie0XHwB0HCPMJ38TQ7LGyv9Ilzsm9waibVF9LxC9kB5PQlFTQ4JWEvnPKo1UZDu4SyeRQA5Dz7gNBtCUv7K9ARQoN4ndSGAAehWwJGLS+FaCLiJMMCC5KLEY31T9E958pJHItIRtJKHZYyiZHdZduowpdBE5GkBT/sr2lwHQv3zvb/p3n/3orevr7/wSfHCUm8RuKLAADVgSMCl+R6ELVnNNhkaRLcB9YHxBK0DVZNBwB5FMZheljQA0NS0TuUgMDaApf2X78wbo3b/87OeSRUsfXPcA/cv3rht96VfRB1e5SezHBc3PmCSQbSxAj2G9IziSHQUq85CgXGErUMVkzMysyrUVgEptssLbJFLCX9n+vAEq02cfXA8A/eD6K788ffLD66/8MfzgKjeJw8iQ85O8QpmqGgQigqNlQXEhX7B4vJ2ulggpBEdcMFn1XABaNlW11DKmSjKA9q9noSKA/tvfXw8A/fNb7eHmX773xX8MPnjKTuIw/uT8dAettxthDO4Nw3jhgW2SEkGUsSRoPKzFBeR8QR1+S6htTCJJfbINElcGNkIQNX9l+/0D9A/X1//hf/QA/cP4+t3gg6f8JEasiIEXDNGoLPg0FkPz9iDSuB0Vh3WPKAzcGDWeq4YbIUgl3TumUnAvvMCG8C2paQRR9le2vwCAfvm/nv7Us/KD6++3r+1n74OngiRGsIioBRjkFIUayrH8ZOb0va1UbRCEiBwcPUf9oAOSrmQdekus6e8l9cHOJfWMINr+yvb7B2ijnpGf/bA/W//zW1/5o/ehK/ZXvYo8AlqgbcNmbx8s5pfmwkS7yfJU9TgG1yW/5dEnMh/IlGkFvUVQXVDUZLpAnRFAKTCGhWl6jhXIKNCY5i2uDmJQgXEYUUiKav6etBmnnLIm0wWqHkC/9Cvvg1+85DA+ghbadhVdoxQswMRn5JPSxaOCURcS5b064SZiIX3QIpy6ROi8U3hJ2fqyc1hdf2V7O4V3jkAHFSQRUiveGE2He5Dx3nDy7INdZFkyQBgj3jh8ApvpIUR5cUXiOoIoBlBNe21/ZXsDaBWAYsiBrUGZ+K3kpiDfnooYlOVCpKaQju6xYlgy8aucTOJIz6woBlBVe21/ZfuLAuhys/AE5AD8/ELgraBWIG8XWZYPAnc4m9y9QUk6X6ixKdO4TirKNIbIDC0pI4iuv7L9ZQF0WPLZrwP9rrdxUn4SCco5HzEK8dYEQ6PTWrcqaAUVz+sE2ARvR4rLlg8hoilRGX4DACi1Tqu0oZyMILr+yvaXBdCl7kQiAOp9hAiDYBOd/buccJ8JD0vw1acyUdeGjUEdr+yiAI12g/IigCZ8ymUE0fVXtr8sgH72w+svj7e/ex88FSQxjbrEfkA2vA8UdAGOCDp9uo13isRVYfPlVAG1k+2ICqAaEoBm9xg3GOiSCdLcZXrB3T9eHEBPn7gPYPqk1tOY5gOUO3Ac3sC77LsG8KGm99Gd+tIucjW4fDl1YPVEKyJb2A4BQPN7LD5mvWCApp5zsoa0s39hAD198qN7ZH7nj+CDo/wkIrYFqEoXcUt3cunTP6cEHUP6AI0ug07vo2ediPvIVGDy5dQiAvCNgAmJ4oS3cjIHoHV67OlyAZp80t4a0s7+ZQA0U9lJvKqtLrD77SSe1tw1wN8Slpjeg6dFSTtJlx7yFe93XASGYGdUy+/K0IColVGo0v4KalwsQNPPel5D2tk3gAJlJ/Gqttq43rezGkDnEJT7DoFojknaD+2NagVd6RsQtjIOVdZdUQ0DqAHUAOorO4lXldWGnYjpfoofOBpMIh2jZfHOexSC61pq//QdQtEcj6Qh3hltdcKMQzf8e3FNSXUGtZ2XAdQAagD1lZ3EkGkz1UZ1iXnkAHp1PPmACEJ5YftvfGwHldo/fYdgNGdjypDaGW11wgxj94SrIIImuiJoUCQDqAHUAOorP4kB0a5iZUwidVE9Yh7dA9JIE8GOR56f43XV2A9IhhAaoGgSiQhI7os2O2FyACpclMQYE7pYgNok0tEAilWQRJ9oMS2P0rXx44D1DzmbDS0z6DqRzbgVkoDckVFi+g6FZb3GBNFQQGoX2u7EyQBotoQxLhegtozJAIpVksSADxThBOqjBufsbPUp+jIAZTGCADq+JbrPeKW3h4EWAqjwmPWCAWoL6Q2gUPlJjIkGBjF1/BhXbsWcs4Ma49sgThSW3hMKtIn+DkHTIAtMOGoXakLYdckk0nISfV0WbM5GCKLmr2xvAAXKTiKNNu/IiYAhwG8rMT+Jw9PONIg6vYdtdEV0g/gOQU++LdCLbgNs1xUYQ6kWV5Xk67JkgzZCEDV/ZXsDKFBuEhG9ABAoflIAPSb4OdbzrsGGB2B+VOcDbqSrdAn3OzSVgp0M3CkvSRuCyPHfi29wXQm+LukczvFfJKrcXttf2d4ACpSZRMALRB4IQ1g9VSWoevQBKlrSFGJ7KEv2LP0dYhPi7WGSyGcXbzshYK42vZH+ukiSOMN/iaAZ9tr+yvYGUKDMJAJeIPKwNMRVuDqj9TEE6NEviPbQ+6iupb9DbELcXUwOE+lFBuMZgFdpvQliA6iuv7K9ARQoM4mQF/GYkV3OZBhGjkN3GpyqcxUBNMAQDE5agu8QqCY+eWXLRR0nOuVUWXGJogFU11/Z3gAKlJdDgnBgUGOuocpTZAlBT4EfHRg1sfsYvA27lvgOhUUCQ3kKJXvZPrVa8yYZA6iuv7K9ARQoL4cYVeBsNUXQsIq7FcR3/oqBHxX+GB6tBdZk6RRA4zJ0FvgUsruj0ldE5W0B1CaRlvRXtjeAAuXlkGJVLH5OHfOTmGeKjlMTLQojBUVBnTgQ4Tet4srLG2wwt58oDSpvDKC2jGlBf2V7AyhQXg5ZuvniCMphApd3y6Va1JeiqqNb9eNA2O8UFChQMoK3CzbIK7E1gC65rmojBFHzV7Y3gAJlJpGGW1aVoDLaBUygZVSUrB7P93gF4k2xXwWAZp7kUl2YtKlJpIX9le21/ZXtDaBAuUmk+EZSAZUOMYT30YQbPhEGuHo4dwTjO+v50eFeDYBmnuT6pZdZxiRtz/R1mZeBUm2EIGr+yvYGUKDsJGJA0VxBrAox5H6G8WOAOtVxjah65BLO0Bz7Z7/fhn2qDNBM+vgNQJVr8FPUovHrMjsFZdoIQdT8le0NoEAFSZTAqhO1OyqMdh7pY8boVDa1auoIXaMrjP4zoUCXMiaRRIWyJf17SVvplBUUdu9kXaZ/Cf+1DQN7bX9lewMoUH4SE6yCMyDBbgpl4XvSIQyAfwY5KEW5jluCx5KCLuXxcwHCCP9eUcMlZdOFvWepLNK/hP/KfqG9tr+yvQEUKDuJFNRi3DB74wEYco0LAYYwIGhsg12nLR5Ajxxxk3kSF8yU7O8V/y1EhdPuBlBVf2V7AyhQbhIx0mJ2sSs00QEaHo8owC0awuD352Ib7DpR2AGobx61JpGmVEFBCCzR3yvOg6x02t0AquqvbG8ABcpNIiJapLYouzNCCDUg261e/Vvvh+LGgug3kCOb2MPdshpARTGgDKCKMoD2r2ehcwYofwEzVLQzKjXujp5c75YoGNlBUZ+f7ol4Zvw4E2gv6mpCxN/LD8Onm2wsEc113+8kkmQlgwG0fz0LnTVA/SuI8T5XREVUxJ/lufL5iefNWUVlfX5OBXLjs/1xduY0thX+ewVhuGyLq0P33S5jEq2lNYD2r2ehsweo8BA02omK9ZtCgIYxMg+NQGmfn+GRZ9EkUlQnlRFO8O/FOYiiwsrIfa8L6WV3cxlA+9ez0DYBGg9+LKdwVMcbftFOMPbHLWCdkR+Dv5RK9kXcaedGoES9sWDsQqVLIvT34jImDsxFm9y1h/BCcYXPEziX7i90Y68BFCkziSTtIBEihri1UEAI0GlDdAofxqcACjdzqCAKdqu4yIhhJexCpksgIUCPuXH5aKP7mRAkV/sCqOhyRIm9ARQoM4nB8AIs4FF1deWeB6OKURD3Y8xPKgz0Z/sS7kIFg4uAOVM0dc7hpQAtFxftTAiSrV0BVHY5osTeAAqUmcRgeMUo4PjV7SPqDBXDKP7nmJ84ItfoxOZjdHg5FQynoTOuhpLT5KIogwygS2hPABX2pcTeAAqUm8RgdAF2xWPP20PUmKqRDu3HmJ/pBqTaFXYy2j5uiBZClqwSitOWDjJINok0S0y0syBIiXY0iWQA7bVRgJ6CUZ8LMI56/twTqHw8+r/KGdYW+IMdR2IaGqzs6fJFdhQLFYx7LZNoGdNM0dHOgiBF2s8yJgNor60C1J/hJVnmyt1M0s+px+HsNOxHlQX+x2jPMYFst2ARQNGEfW6IQaKF9HNFRjsLgpRJQpyz6L4BtNdmAeqe5CVQ6L56rOKr+fL3BNPgQd0wTGBKdApUIspHAIWxUlt4D1Y5f6/86En3cyDIgvba/qJSNonUabsATTEwwFlTDSCOrhYK0A3HIea4whA4/lSAaUsIUL6txJZECFYZf6+S8Cn3syDIcvba/rJitoyp1dkD1EMEYhxdOpCzeSzlVHCqUqbs0VjkTTclmEQiQ3FbUh6s5H+vQgPe/TwIspi9tr+wnC2kb7QfgCaoGew8MrST1PfDjB+8+qlJI/8T+A4RBWjH5FIpocR/L9p4huxXOXX9le0NoEAFSYTA4pS6VgoLA3MiQDosrh+H9Zy47xBRQBQ75cFq+wCtbur6LxNWbK/tr2xvAAUqSWLIupS4lUeD4rKRt1+RD+6XB/VdB9KR/A4RokPXQsrmAbqAq+O/SFS5vba/sr0BFKgoiRQO8zVFHj6TIz+sOBai4gaRcKn2pf8ETInvEKG47aA3QiuiAdKCZBrnKP11WcR28l8iaIa9tr+yvQEUqCyJmFloKy83tEM0OASjimOZuAXHYDvR5qOHTuiKv0OU4qZHW4jeC7X1STreKM4AACAASURBVKRZnUv7LxE0w17bX9neAApUlETMwtQGZxczyMghGBsGu25vjwEEvU9MO6KFpcDa+w6RiqsHW6guCLX1ZUxz+ibwXyJohr22v7K9ARSoJImQiWCKBpVL8oPcz1VvPnaLN/wd7qdEU5imTZtkBGG2SDLAaesL6Wd0TeK/RNAMe21/ZXsDKFBBEgUAasuyxKLHGLmfqy+5/SKuT7YfHN+2m2YPIVEGGOmOIZtE0vVXtjeAAuUnEaOQAyUpaEHvpuvLbgDOajRsj5svsgPcXi4BfMBO6O8lqVdHtoxJ11/Z3gAKlJ9ESBsORQJIuWJ2k/WFT1DwmgviwdjuJidfTA+YvUwC+IC9wN9LVK+ObCG9rr+yvQEUKD+JAjDmKB5wiC5TAyBhMx5B49Xj2hWXcYcQ20jhWlbATyLg2LP478U3RKCMG/+MILr+yvYGUKD8JGbA0WXF/X/Mk5A9E44IcwEaVBS0yN0U/aTHrJUEoipT9/ruC62kynn0hBFE11/Z3gAKVJBEGjqMUtU8F4YH/p1Qo/L4KSEorFMVoLIqfueqAzTr4WdGEF1/ZXsDKFBJEnkUkkBK7Y/PrsEHZx3q0dudxU/Ex6kNZImKABVX8Q+vawM07/G7RhBdf2V7AyhQURJ5FGKlqsFl7/GH6U6oqVHdbukF0AA3ZBtRNQOoqrTttf2V7Q2gQCVJZEHoA8l5nyrrgQDWahsQtymDIDFucBuicMNGZhIJRc1oK72HBejMSaQaAC13zw21EYKo+SvbG0CBCpLI0c97f6x9rHrkCCIZxVFZvhnpemER/On/b+9sf2U5jvN+aUnhiaQEUqCjSEGgfDCgwCQiiw4QGjEQCAihhIEEBI6voxAyIBBk7L1kFDImuf9+zu7OS3fXU9UvM7PVs/M8H3h2pqvrqe7d/nF2p3evyKTXBhpsgC7bxrQCQBf516XqhCBu/s72BChQ/STmkBfRxQBUkkLJlvoY72FrLkFBb+mFTHAanFitQK8NNlg3kcyMBVp8E6li8nPKpuqEIG7+zvYEKFD1JCqwwRBS2KRjy8oGAZo0Z5VEWkVAF3QIz+TLKG8wtjEttFu8jal24IbyqTohiJu/sz0BClQ9iXXIw9G59g0BCu8/QSfsgg7hmWwRNe/wjY30i+yi3HkRoL7+zvYEKFD1JFYyzw5VPg/QLmgvBcwLDHZUNFJCxKlGJxh2TjrJNCWzqMZmkzStoZrSTK0FUHuOCVDN39meAAWqnkRInAqC4o9LxeJRTE7po7IFPL5PlZHaaFLdGsTvScs0BZOoBueztKyhqtpMrQRQJX6b0a8oAnT4uwv1ClAdObBBBtqdJqfrgWhDafLrd7xTgmKVwQhdG+J/ldO6iWRJdcqPpjuANlzeqj2yqTohiJu/sz0BClQ9iQA4EGj4m+/6DW28dnImFhumlnmvDoouSZW+htRL2aJJVK3yNfQH0OptTEY5uVSdEMTN39meAAWqnUSNX/LUGxBxym3FFIZxo+IOV93cZAM0TVryGkIf4+X6IsPShrmAMofKrKVaZSN97jmz/CtsNhABOvzdhfoEqE6wVDOt7C2V8FzUHDSqbnGupNIMQMPozPAL5yuXqb2MFW4iFQxTdV9jCaMnrdR/BfsFIkCHv7tQlwBVARbQyqLbCa4fERq3B62aaZo+rrQAoIVvRMvmK59Ljch1Xb6NqWygijsB6urvbE+AAtXNoQawCFcG3k5o/YjIJCBsn5uwgQi8lZTlZ+GFWfFPsueyqe2Zjos30pcUp7uvsoTbS+iEIG7+zvYEKFDdHEpcpsrc1X7zRjSIUCuZ2MZUVN8p5qfsVzLu8DVUNkd1HkWqeb6iGQ7OtRe3EkGaC0jsN5pj3d6bYM72BChQ3RwqgJKEi89NSigmNzLlTNKN9KX1pfysW3lTjz0BdKohLGZZcWsRpNlfZLkrQQnQ4e8u1CVAlbfIGU29xXVgfFv+qQSgcS1KeVVVJYfGoPsGaGw5FRFVs6y4rgiy4Sxr9j0N38GeAAWqncTxRduCKvlGWmS6hJmp5oLR8qmpKn1Xq445dO8YoLGnMt5lxfVEkE2nWbHvaPge9gQoUPUkyqVYCKppQ3vaKGKtVNFGTLl8UB+7LlDnScu53k2kVun+ias23kXF9USQ/PO2vn1Hw/ewJ0CBGiexhZ/iG5Vju7rKQSrpHxcIzKMTb5KOuFIlpbmRfj5tZrN656U+X+ko8Oyd0gvVSnfvJRw8zj5tG9h3NHwPewIUqHESNcRpy/YqsR1ziBDhRq4cQE+iR7Il1LBSxhwEGF/lLM1m9C7QcoAuuXfdE0GKJnpl+46G72FPgAK1TWK8LMFixa/u+efPzXirLQtQ0DYeoc2gaq0gn/z/h2pspVvyNroYoMpNpGXqiiArjqvUvqfhO9gToED1kyixhnGHXt3oArQ4262AIZPBKdl0O4ovf0VwdnfUCfz/QzFWKsuWnlU5QPE2prRLpbv3Eo6OWuew3X794Vf8mjUBWqdeAQq4hnkHX91L+Zm7iRS2JDuVToif6cW0sizHU/0CFMxHNCStQ5279xKOD9umcIH96sOv+fdUCNA6dQrQCubFPUXoCTy0375fC4gSjj2E09xpbo7gbYXrC3PYhKCM0Z4AxVAJUE4bP4dkptSrKO5w6g6gd7df23/+SKvMf2X7ShGgSJWTCNGWYZ7WMYjMpZqziY300ivqEzTHF79WuKh+eg1VjTE7h1Z76Wnj/ySWslUiEaCrav6Rm0L/de1rRYAiVU4iBhtkUKZfeN05d0FxUbq0YOSmFaN8GV4dSX74ZnN2Cu32stOtN1OyZSIRoKuKAN1SewGo3IU0Pgr+C7rVKihg/hNxYI6aeqDC4nxReO6mPK4pqU7vnoRUtavdtAajhJIyoAjQVUWAbqndAFQi5fYgF1qp0B8SMg2LyxW1p6MC4zNGj2g9ZVJ6ixw17Wo3M16poaAMKAJ0VRGgW2rvAM1FVgraax8pgHIxP4MOw0M91wlTp+QMnsDcDJec1hpsj7IIKQI0VNUWJCjeRNpQXQJU8ipRUVCLyuynSFGvys8odeqhDt84pfZOR5Cb4pLTSkPOJBl8qQjQQHVbkLC4jWk77RKgJTGgV0lMcWpYMARjcAQeilwn63KzDKBWbhGTszYa8i6nyrv2VxGgsyqvHhVxI/1m2iNAC0Jwr6KgitwzHMbeoDU5IR4+Sb6cxWkZq/SGA7InGDWpfZQ02KZRWwG0FCOdEOSi2s8vV/G/nxW0J0CB6uZQA1awWO0QrVtRUG3uqU9u94CaWLyGxGkZi3vnMqMo0GhspMd5sE+jNgJoMYg6IchFBGjv6hKgNsJARO6rmxA8epQ4mes0dM2VkClNDN88hXpnM+PhCZU+X7ZPq7YBaDmJOiHIRQRo7+oSoDkQiRD9x0PkW8+wEee2tjFZZTUGGMMXJ2WYOW9FswtCip8vw6ZdmwC0AkWdEOQiArR39QjQLIhkkEbQMRKmR1ZRBLqybL/7r4zMGn96VsaZE1c0vSik/PnSbdpFgM66Pz97Gf59wLRYHQK0hEQiDBF0jlOSg7PyCPTJVhj3CHLBdn0CaqZNZi+KgiGL1lBz+ZM7ATrr7vzsZfj3AdNi7RegcdztZQYC46WswAN1mQOUnKWKGH59iMejVNM8c0VROGTJGiopwRYBGure/Oxl+PcB02J1DFDtc82AR/PJYWlAdo3xAG2pZ5I1dWm5TW8NUQmxuhZN3dKo1jVUNPisHvAmUtVGdm+COdsToEA1MzisvTfK55rBNWVwFgM0Wsom28ITaYDooXpAX2OMSpDcxlQ3d4ujGtdQ8fBtPd42pqr34QTo8HcX6hCgt3U4XjCokEL8NAhq8zP+qDQNSHskWYJWcd4co4aZ+V91rlZZx1xU2xqC096gh9tIX3cniAAd/u5CPQL09DTzM7u/86r55VkSnV/baUByHOSY/iOuiYuGiWO9l1DbGqqdZN0d2N/zk8CtfhG+cATez763PQEKVDmJLQAdOhbL9k8i4kORamyuQ4ca672E+gPoXe9FE6Cu9gQoUPUkxgAtAuNJIVvyKFza0RKfDpTVH5xRMAHOmRDRKBP8o6SVE7eSVgNom7uwr3sPvFQEqKs9AQpUPYkNAJVR6XvtdG0D+p3Ut+EgOAlUT6njVJrDf5S0cubW0VoAbXRP7SsJtFQEqKs9AQpUPYnJTaSyS1CxfPHjcW1D+mlvw+NTkBT6GXWguPEsEt1ZK91EanV/MIDyJlKVPQEKVDuJT/o2pgrFnzKmizt8rPqMFSVnYJhxYjosGj4uQQ1tbdVlP18qC4oqzkukKAboCuYnbmNytidAgWon8bISAD9rvwMUUyjBUvDYAPVQUXICRoleyL5k+LACI7Kt1ZD5fBk0KCk5K5miFKArmF/EjfSu9gQoUO0kajRTOQeb5X3u8HhuNK90k47wPXyS/ZS6iQ8MbMHkVmBLqyXr+TJZ1m5ppqjh53KCdkIQN39newIUqHYSdZ6ZStGTUig6ng/sjwpkz/iE2CCaHN5OgWhVwD8TV9RaZn6R8XyZV4M1g1SEU9TwczFBOyGIm7+zPQEKVD2JBtAMCa4BhIxn5oeZD1vjrlONcL2Cw6SSkuVdehPJDsIjyLufOgRoyXvgFcxv6oQgbv7O9gQoUP0k6kAzdOk7H0gM4U6Z21WnuO9JOZ5OpsfpB7Alw9ejUQlFAC23n9YQCncC6KY9E3VCEDd/Z3sCFKhhEq8rQaUa1knjp3Lv59bwBhAU7AaND+FylesXWBcNXwuGFZUAtMZ/WEMwnADdWgTo8HcX6hegNyWwewKC75SL9rqP5yckTOfOaD+9kmku21zAFatbny+rIjsY1asXYCT3uIm0dc9YnRDEzd/ZngAFaplEADukiA12IAbo5T8jEsbIGKAIQUnjCRE3ltF0it8uq/MlHLSc4cQoXS3F0580mjd0ii10VadInqFF5hd1QpDlavvqQSfDvw+YFqtfgALWQS0F6E1vgl8juRQQR57ivvLoBIArpLcka78coEpOUEuuuEgmQO11WehgqTJF8hQtND91Q5DFqtq+H/ivZN8oAhSpfhKD9atSr06XxGZrsPzOolFcYUZtJwDcGsUdKwAq28UooUXSI5EN0K3V9g8YrOi/Xqom+/M6v95X+O0t6b/UeJkIUKTqSdRJ16xLYqs5XIegq7isC+Mxs4qV9GwFaDQQEYs6Ksl2BNANyvQnSOO1Y6TSr29J/2W+S0WAIlVPokG6Rt0ypyeVuGzX8cRYMYivUNK1+CaSlgWVj642tXTWTaTtdXiAtl47RiJA76LDAHRMnZxUQkVXNR9olONBZ3FR0WsISHNAJer1YNtZxjamO+joAG1GXyQC9C56dIBKDKDWtFOSAoWdcMYnjZ/mm+ekszVf2AHXqBaEbWcZG+nvIAKUACVAY1VPokqDasnsUVtRbzOtajceokqic2nvdQFq54IxvmvogW8ilfAsAOgqBK3tR4DWqFOAZrBWpTR50jQ8svqaWVWz8QRqjM6JgKVv4Qu2VMkqQu0JoOt/0rDd6IuIFgJ0BYJWdyNAa9QnQJ8UGLRJTT4fn0xKmjk1KzAApQYRsPAm0jQmNTKXblcAXf2Ths1GX0bF81qXoNxIfwfdCaB1QixbIi17dAq5quXY6bQxKJ20Qq2Jic9VJZEJa3rsSL0NbcbiepGUs7q8AkXwWSIte3ymbEuTSJg/DdqjcyKgZh+oqL1+quXZfV2BKmqZjpv/KvZShdeV0z7QxVegbeIVaI12AdClSFXSy2PUz06mmgUn4geghjR15Vc59ZHWakzxCABtnxJ/gN7533FO/D1MA3sCFKhuDiWUJKdqhPNLN/SBpp1Nd5NZT6AdFlT9YyLGUKs0pXgAgC6Ykg4Ausp3kRpFgNaoS4DKqz4EqmLh/MAs3SSKSonyWX5p2hNoxwVV3EQqGGu55hwE6CYqvYk0xm5VR8bfx3ayJ0CB6uZQYgGTqkynzH1ay8b+LNNwPIklDPrnXkNqscZMtStIQoBuo7JtTN4Ec7YnQIHq5lBiQUMVkPinkHO3E8xsBOj99bgALdtI700wZ3sCFKhuDosopkj+W/LZ2wlmPus9vNonaTV225/Q5fHwa34gRGRQKmvZHBlkeQCARlNV9364E4K4+TvbE6BAdXNoEi0jCVANXIVuxm0k4R216ymAffIaQj4neFkVOZhZy2f9IQAaTEHlHZlOCOLm72xPgAJVTqKNNEuXlZLASgFX6qWZpreA0E2k05vBO2rXMmP78DUUn7SHMhwphbXMuvcaWosg4wTU7gnqhCBu/s72BChQ7SQqLFMYF4YkBD3lAJrPmIbNK3M+N21QeQKXgTChsFdPBkf2UDJZ7fCgXu81tDJBqneld0IQN39newIUqHoSMcss0I2rXxDUpklBxiQMvaMOAJomN/nZBUBjq+v8l3TbTASor7+zPQEK1DSJgGUK40Je6J2gFcgCdoPCMARQywAUgRvik8VD0RIUBgenCFBHEaDD312oY4CG2LldVJ4wQZXTKXaumQVUwkCcOw1DQms09JYJRVZ4WhyBJECKXXEsAeooAnT4uwv1C9CUUHI7ZbjuDbidxDagyCsMxLmjMIxwhZ/lb+DLbiKdwP8BoBQ/M/RRAcqbSJX+zvYEKFD9JKaMeiHorU0nEkIbhKD4nHI+m0QkD5VPEU5gp0yuFuAdDD8+CfumKhiq2i8NfSyAchtTnb+zPQEKVD2Jkj0jQfN41CIwyKKTSUQULEKi0BZ+6gAFOMxMmTrWTL8jAJQb6av8ne0JUKDqSZTwwW/ig06YWZCf8BI0jklbFBdtBEYdmnc4/PwUZRLlK7RqeDSAVvo723v7O9sToEDVkygBNO+xVLgku0TtOs2CU1FENouBpyDCDISZxBXoHKucEmnyFaY1hGcIUE97b39newIUqHYSAQmDTeoFhJT4MLg3n4kiUAc9v0w5RpgYQ6nSz0DDSHxKZMlXCMc/FlDQaTuRIL7+zvYEKFDlJCIUvklpJAgBOik5T1Nzyo6wI8qk59eGUD7a6cQZWRhDQqWW2wO5rqG1fwmzehY6IYibv7M9AQpUN4eAhDH54iCjH05qYfB2BmzkRw5FYygfLR6Hmk8dUNCaM8fyXENr/xZ7/Tx0QhA3f2d7AhSobg6fNIXXVYhQIh5mtbZ7DhcsCkCt9OoYykerDEPLF55SeuS8FTmuodpNmzmVPguBOiGIm7+zPQEKVDeHAnDjIsDIUPvCtLm3+rdnT1aRnikeQvlotfHjKP0D22ZwjhLPV3nOhe7ga0OLMrZMSScEcfN3tidAgermEBDutgSCxxotLIpA1sDQ/QG0pris0uerPOlSewnQZRlbpqQTgrj5O9sToECVkwgQd10D+KzWNT4xtKe/F4pXlwRoWhaoGJdRPtrpIN2EgMaBTtWYmjpHs13xNnixvwConTHr1TIlnRDEzd/ZngAFqp1EwDhNSde0KQl80whQkVvZTyTLKBiusEjGr3iIU1Wm+YLy06T3a3VOAWpnzHu1VNQJQdz8ne0JUKDKSdQWcMmajtuS0PFHO3NJ0ptIoC7kaZ+yxxv3Qk4g3RJTu5yyyVY7NnsrF6AwY4lZQ0GdEMTN39meAAWqm0N1ARet6bAtiZ1/9jjNkaQ7B5mUymC10wnj6khfzkmi+qVf3yNTRnaytZ7t7ugTUJyxzG2IqCiqE4K4+TvbE6BAdXOor+CSNR22JbFvIEFPci2ew1RaclBs2ElZsvqSxzXVz1tFDytN0WRrPRfYi1vwWsZCt2tATVWdEMTN39meAAWqm0N9BRctaQmh8cTlPwKiJ4Gu64soPqUai7OZRa03J4kyeapS1+eASuKWl4BDo5eLkVFMu21UXFcnBHHzd7YnQIHq5tBcxdqSFv21TOIiFNxvT36PtJSfT9nLIr05SZQbY03qcqH5QmlVn6oClOD45aJnlNOeH1ZJWZ0QxM3f2Z4ABaqcRHMd4yUt+uuZ8I2kKGluG4BS6WktgLbgsL6HkSIZV5LVMKrw17IkLxc9o5ytjBMBWuLvbE+AAlVOoraQ5cqGXXOp3pgOJQXg7PGZzMiKx1w/aaU9SktY38nMUv5yGbvnCqoruBOCuPk72xOgQJWTqMIkC5f0dBAa9crkLPQFJ43a7ObyMWYnrbRHYQkbOJlZKl4uQ+9cQXUFd0IQN39newIUqHIS5RK21/Z0mK6T8DjqpeTLNcb5QSWn8ER4HlVkD1qPzMxacYe5G64gZ+QP0NKCpuaS3ynphCBu/s72BChQ5SSChWzRRaVOdBy1KukK+SltReXTYzy2glGL3IXTVhyedir2bSitJssGAB1HWfRLT50QxM3f2Z4ABaqdRLCQdbro0ImPo+bxkbUnVJWoaCwc5UkHh86ZQ6wjaHFw7Bm5nzOJqguryrLCPwmFIwp/K68Tgrj5O9sToEDVkxjQI8M1peGaJz6Mm29/kxvyhs+kaQXq2dtuoy8jaJ2T6ZhfQ7UDq8rSQpCigsBv5SF1QhA3f2d7AhSodhIBQlS4GNDRji7HV26mO5pOeYDOK9Dq1vg+1xhLUc/Kedau3vPPV5NZYZYmgpQURIAW+TvbE6BAlZOIVrUGNPOyLSJQ3DivJtwF5L4chksw6CbN46zLRl/eZ7lXGUC31GYEIUCL/J3tCVCgujlEq1qVvf8nxErcBgCaRoFS4jU4nTLLqnwFtWRYbBbPzT3XkCyaAPX1d7YnQIHq5hCtauOzTtwQ8FDmvRxNiylhD+o/nUvW4HTGqLfyBQQGVjNjlW7gJtLprmsIVL0dQXgTqcTf2Z4ABaqbQwWHJiQV4by3w4F7J7wVEpcUATQ8AaybiAYGVjVj5cmjc/Gjc9RaN4A6oYo2JAi3MRX4O9sToEBVU6iiUOUj7pGuTHHyjSCdBqLpdAjQ+BhYN9AnKaA0g1Z3SdR8PD06xz1qBlAnWNGWBCngZy8EcfN3tidAgaqmUCehjkfcRwvBnYLrobTgOSjmpw3QBrX2LulVXts4/CUjKRIsiATx9Xe2J0CBamZQByE+a/Z6grS0O7xIB+hTxE8ToFF91YOvvH7N21UUdk57lNdRJ+iwJ4KUXNHW2u9o+FvYE6BANTOoMM34DNToNcckZ4z4JxOg40b6+cAC6NK74nU9K1KboQRooYo+U62138/wN7EnQIFqZlAjUYZPSrcpKD42oiUxUOv4MHlLX5CpbvQ1M1eT2QxdCtBcl5k6yGA/BCm7q19rv5vhb2NPgALVzGCObRoIMkHRUU1e+MnB/DDDT5Cpbvg1U1eR2AxdCNBcn5A6IHY3BEn2tK1lv5fhb2RPgAJVTWGebhae1KDwQImCiWFo8Dh6S582PiHE54e/ybtnXBTSsptIuV4xdWTkbghCgG5hT4AC1c0hxJuioo5Jm/5xKkiNE8oCkjPRYWqafw0Vx+bmEU8Pjp9QsGgbU4mLiZ3dEIQA3cKeAAWqm0PILEgvdb9m0iNFoxKECJotQViLQ9En/xoqjs1OI5oeHD+zYNFG+lzpBGjGfi/D38ieAAWqmkKIrIRYEcnGZR4t2rhDmkLLKwmKeon0UUlxL2P/v/Ua0kILuqN6CjTBIKy6tDPwfXyA8ibSFvYEKFDNDCK2Sf4kSJsfB3neTGdkCjtvcCY6CiySfiel0d69aryGcGRRf1FCiZT9WGWda4wfCKDcxrSBPQEKVDODCG1gUV8fqK0XDTB4g3LC/tI+OjG5pnWGlcs2o0LzNYQiSzPA8mxhfi4haM5Ja98TQdbn566Gv4U9AQpUM4MSOCqCzMbwSseCWHoujdJ4YDEmaHsSj7MzoALUsBQnrfKQFH7i7nbanG3muo0E8fV3tidAgWpmECBRW9VWWwag4O45wPLJuJ6yEKMVZoNllPoWXreUZ63ykGoAmsubc7Wv20gQX39newIUqGoKp5WLMRSsXqMp+axNDQOkiM7IaFxn0miWXPIaSouSlcEpy5yTmiPKAVqWuVkkiK+/sz0BClQ3h4Jew7FY1whTAKBzKvX9+RMAY2wZFRbVGf8RSaL8ReRRtzHhs8p5GKlO9Thh6nTmq1hLJIivv7M9AQpUOYkJbxKACR4pCz64LTLnVDMm9iL8BK68rgeKfdqWZrZfQwqklP4ousgqDpo2McHpNN3WFAni6+9sT4ACtU1iupbFFWHmFnf6zx2dJIKDoyAMtmevCkVhoksheoyN9Lg3iC6ygkHqdFalXqA7EgQOohOCuPk72xOgQE2TKND0JL/No+DrlLbKM/hDgCBMZMAmuAJcbiF6rG8iwc4gushKH5H1YUPhKJp1P4LgUXRCEDd/Z3sCFKhlEgWZJKaSmPTmrggv+BDgduYsc1QBdHqsl5sZvhEYnwQ2sZVhpAWFPyZidDMyL9HdCKKMoxOCuPk72xOgQA2TKMAkKIX4GRIU4UomsfJHzfCGEwao6KS6668hNRANSF5LlfjkAKr2zyZepHsRJDN6LxGgw99daKcAHeIiUIkvuEBYySymg/Ie3iAo6oRj7deQFoeLlxeLWRcj6Gw1jk1m4iUiQH39ne0JUKCGScRci1/u4bk3gWQr6GJegSJG2wQNL8u0pVlIUAnQOTrubeVSPcLTOCj5CFqrcyMRoL7+zvYEKFDDJGpgw9dzMUDfJM1BapEo41ICrOF4Pm2wBw4EDB9c+hbXE5eFKzCDCFBHEaDD311ozwCNz73RCRqmFnlKXZJgewBWVD5HvAkhzhb3Lq2nsLSpgMbc64g3kXz9ne0JUKD6SdS49gSuDZ8kQEOCRqntNNIlTqJCJT1lkCcPpvP52joGRR3i3vlc9e6nXgG6fil4eJ0QxM3f2Z4ABaqfRI1r+h1tAFC05hRAprRKeqi9w9PpGWV0WS6dh3aTmYW5NPM8QOtzryTl5bJFMTBjJwRx83e2J0CBqicRUXN6JL6zfesFABpKpAkbcrdVlODoZPHqPUYc+gAAEUNJREFUnoKVPkm6ZJxV5aq5CwC6CbIKhF8u98P5nQii/SQVATr83YU6BWjKzxl9J/SzQUM3E6AiTdwUHxkFG/ysIuhJBVSaDh2b9eSsSwG67W4l3R3NfvUcL/Df3OEi7f/zBOj4dxfqE6CSn8G6CX4jJGl7AptB05SFz2JjuU3dsulWRUdJMt81dAiA6u+UCNDh7y7UJUBjco6wHBqHo4igQb8MPwtXYabgxCJKW7zG1WJkw6rkKEhGgG4t460SATr83YX6B+h8uXlrHMGpAPTGWzNjwTK0C05f+WHW8kVeAdB130svHf7WIkBd5W1PgALVzaGgXUjQiZvhb9UNrdoak/wMOsFn0SpP7JRStmwWj1JrKUiyjToE6IPdRCJAVXsCFKhuDiXtAoJqKDRuyyB+2uvRKjh67Q9JUn6WrHM99lyaYiv1CND77QkgQF3tCVCgyklEBFUBGl2cwDVmARSvyFKATkkC++KFroYOG+n91CVA77YngDeRXO0JUKDaSQQE1dEXnMFrzATo3CXobO3jCQAqajFSypLUayrvJXQ6J0O6s7v3Er6LC7cxKfYEKFD1JLYBVCFSFqBig721k7wCoBohw3f+cPg1k7WBSoawnbyHz430rvYEKFDTJBZA7wTjkmx5fkYfX14LiPol+QA/BczlAarIHL6fgvIylSrS2FAm7+F723v7O9sToEBNk5hD3iCrLZ8G945/Tw4T1KomMU76qw3Ja8hJQXm5SrHUd6dlIkF8/Z3tCVCgtknMUG6Q3aokMZNfX0RKumtKnZ/hZ4Z6/wyWdg5Q/f5ImUgQX39newIUqHESdeIFoFIYeBIpkjbYb4g4p26hDDclTm3AffcNUGOHTplIEF9/Z3sCFKhxEiOCxtiZlzVGoExQ1DZEpD+nF1WnZzwpcWoD7kyAusrb3tvf2Z4ABWqcRJWg0c0NGJD/7DPKn7xlF79HOtemZQqa01AxPr17MnxNsNtqApNY0ZsAXWjv7e9sT4ACNU7iSSFocmUklnvmM07Jh6SDBKh12RpWjoLBAPUEyfCx1LwrKchfb0WALrT39ne2J0CBGifxogwUp5gouHrvUvL4LBKIaiD+0nMqfNQM6fCtzvmJbNSyjfS8ibTM3tvf2Z4ABWqZxOjb5hYFQ0kaxrHJMW4S25gmm0wFWlVA+hCyS6jCpVEL1xC3MS2y9/Z3tidAgRomMf26uaak63BKi02O8eN0I/1sk6mgAm3GEHYPUG6kX2Tv7e9sT4AC1U+i/MEORbcew6OUirm+6cF1g+cp/i58agMqGAOSzOYY09TgNVTfcyX1+WMid/N3tvf2d7YnQIGqJxH95JECwYsAD1s0vfesA+gUAWvTpAcRoL7+zvbe/s72BChQ9SRGt3It5l3jqyipa7YU26jSY6UUWJ0mNcT/JtJmmYvcvZews723v7M9AQpUPYnJXpgUSzGhbKgV8zXwTLdRnZJj1Qz625Jh7tuYDP9tjW/uVS+X9QvqhCBu/s72BChQ9SRaAA1OnGQrAJjSlJ6AABULNGOWZM6PFcR5b6Q31tDW6L6617xcNiioE4K4+TvbE6BA9ZNoEFSEZ/ipfaGzDKBItlvT3fhk+L5S/cv/v7DEvWL4WxTUCUHc/J3tCVCghknUbyOFkdfjHD8L38SXAPSWdT2AwkivJTSVofmXj2uJKoa/SUGdEMTN39meAAVqmMQAZwqmTrm77yLQxB+8iZTITAHssiPtCaBzHQSoowjQ4e8utAeA5vg5byTS4tKVBjIGBD2pL6K5R9QfGJau6o4AGhRCgDqKAB3+7kK7A2gYljDsdNIJGp0G/BzOjZ+54oLDPuF/0T2dwkXdD0DDSghQRxGgw99dqF+AniQ/0zDUokWHZ2FMdCoP0LWEMnYLUN5EuoMI0OHvLtQxQMdvw68D0PBNNY4Jz9wPoG3bmDZQEUC5jWl7EaDD312oZ4Ce0lvwaVgVQIM31UpMcOKOAG3ZSL+FygDKjfSbiwAd/u5CXQN0kE4t/ULSXlTZmOxNpE3V7U2k+4gE8fV3tidAgRoncZBOLdRSwrhcTG4b07bqdhvTfUSC+Po72xOgQI2TOEqnlv1eXVcmJrORfmN1u5H+PiJBfP2d7QlQoMZJdNOxCXLw4Xvbe/s72xOgQI2T6KZjE+Tgw/e29/Z3tidAgRon0U3HJsjBh+9t7+3vbE+AAjVOopuOTZCDD9/b3tvf2Z4ABWqcRDcdmyAHH763vbe/sz0BCtQ4iW46NkEOPnxve29/Z3sCFKhxEt10bIIcfPje9t7+zvYEKFDjJLrp2AQ5+PC97b39ne0JUKDGSXTTsQly8OF723v7O9sToECNk+imYxPk4MP3tvf2d7YnQIEaJ9FNxybIwYfvbe/t72xPgAI1TqKbjk2Qgw/f297b39meAAVqnEQ3HZsgBx++t723v7P9oQH6za/ffX7+xe/E+cZJdNOxCXLw4Xvbe/s72x8ZoF+993zRv/xfaUPjJLrp2AQ5+PC97b39ne2PDNDXzz/93fnLD55/+g9JQ+MkuunYBDn48L3tvf2d7Q8M0C/evV57fvXej/9r0tI4iW46NkEOPnxve29/Z/sDA/TT5381/P1l0tI4iW46NkEOPnxve29/Z/sDA/T18/vXv58PIJ3VOIluOjZBDj58b3tvf2f74wL0mw+Gt+5fvDt+CPpPB63oQlEU1YkIUIqiqEZtBNB0I1PjZbybjv0e9uDD97b39ne251v44Ap0VOMkuunYBDn48L3tvf2d7QlQAnSpO4fv6u9s7+3vbH9cgPIu/FruHL6rv7O9t7+z/YEBOu7/5D7Qhe4cvqu/s723v7P9gQHKbyKt5M7hu/o723v7O9sfGKDffPD8E34XfgV3Dt/V39ne29/Z/sAAPX/JX2NaxZ3Dd/V3tvf2d7Y/MkDPX/76hZ+/SK8/CdBKdw7f1d/Z3tvf2f7QANXUOIluOjZBDj58b3tvf2d7AnS5Dv7dTw7/yOLw9yQCtENx+EcWh78nEaAdisM/sjj8PYkA7VAc/pHF4e9JBGiH4vCPLA5/TyJAOxSHf2Rx+HsSAdqhOPwji8PfkwjQDsXhH1kc/p7UJ0ApiqJ2IAKUoiiqUQQoRVFUowhQiqKoRhGgFEVRjSJAKYqiGkWAUhRFNYoApSiKalSHAP3m1+8+P//id95l3E9f3f4hlOFfQomG//hz8dV74z+BrQ/8gWdhGv4BXwR//++fn3+cf8L7Hn5/AP1K+3eVHlZfvBusnWj4B5iL188xQcDAH3kWpuEf70Xwt7cB3/4F390++/0B9PXzT/G/7Pmw+nxcRBdFw3/4ufjm9fM4eH3gjzsLwfAP9yL4/PnH/+F8GdcVjbt99rsDqP5vyz+sXj//cnocDf/h5+Lv/+J5JIg+8MedhWD4h3sRfPPB8/uXvy8XmO/v+dnvDqCfDq+oT4MX1GPrmw+C10Y0/Eefi0+fn//8f09j1Ab+sLMQDv9wL4Kv3hvelL+2n/Deh98dQF8/v3/9G72leWh99d5P/+fLpci/u35MHg3/0efi05/8l2ls+sAfdhbC4R/3RXAF6H6f/d4AOv2f+It3O/3QY3WNtw+e30+Gf4i5GBaGPvDHnoXPp08wDvoiuL433/GzT4C66/OXd3L/cP5/v342X0meFW4pAnT4e9AXwfUt+o6f/Y4B2uvGhbX16fQm9pfx8A8xFxKgycAfexY+Tz4CPtqL4PPrNqYdP/sdA7TP/+Nsp8+f9/Z/3zXEK9Dk+FjDf/fH7593/ewToN0o+R/uDl48a4gAjY6P9SL4dNhGv+NnvzeAdn/XbTtdXyK7ugO5hg58F/4iCdADvQj+9nncu7XfZ787gI77vXrd97W6xh3Ft5dINPwjzMX8IaA28IeehekC/Pn9+fgYw//m9fNPxo819/vsdwfQ3r95sL5eh2toV9/CWEPzPp69fhdlkeYL8MO9CF4H387c77PfHUBfXkI/6fm7r+vri3cvO1i+/IvriKPhH2EuRoLoA3/oWQj2gR7sRfBpOKD9PvvdAfT8Zd+/vrKBPh1+h+f6LZRo+AeYi+mzLX3gjzwL0/CP9iIYf75v+DWA3T77/QH0/OWvX+brF33+/2YbfXn5YcQ/H0YcDf/x52K+OaAP/IFnIRj+sV4Enz9HAN3ts98hQCmKovYhApSiKKpRBChFUVSjCFCKoqhGEaAURVGNIkApiqIaRYBSFEU1igClKIpqFAFKURTVKAKUoiiqUQQoRVFUowhQiqKoRhGgFEVRjSJAD60PX8X60V08v/NJZZev//JVfSeK2l4E6KF1V4D+/p9/cvOsZuH/+e6rV2/95+1KoqhGEaCH1j0BOoKzAaAfvfr2z169vV1JFNUoApS6vEPegE6p2ml1qe+jV3/yN+vWcyZAqcUiQKnuAfryDv6dz169emfdes4EKLVYBCjVPUA/fLn6/Md/vQHsCFBqoQhQKgboy8E7f/ezV6++dblp84f/+N1Xr159/+dXzLww7Edf//Z7L00/H7Dzx2vzP/vV2DcKH9tvzR9NH7JO1Lq2vvWD/3HGySe9NL19STDeRrqWeynxrR++hP7dv3lJOyQBOW+nbx8AJB4f3eHGGfXgIkApAdA/vXDlhThf/7fx5tK3L58/vvDnX/wsOD7/PmpOwydCvfrhGQF0bH3r355R8lm3d+8v7+N/NFX4T4a7X3/yN4PnAFeRMwVo6EGAUotFgFICoBcc/fFXV8JcrvH++AKdS/sLf17I9Mn5j395o84L0r7zcqn39W9uzWn45fil/YLVK6OSu/Avrd/+65eL1p/dPtxMkwf68Eq/l7qGK9dLhRenS+ZvvfrBx9cKlJwpQGMPvoWnFooApSRAb7drbm+dbw8uoLnw550h5HI83Rj/cMRTGn7F00v0NS4G6PSZ5tCcJp815v1orOvrkX+XPm/HFaQ5BUAjDwKUWigClBIAHbj42fjGeObRwJsbrz6KN7en4dN986EhBujc+fbePE0+a4x8iXs7qfDD+a375QHImQI09iBAqYUiQCkBUEGV5BJz4NFnl9s1fw3yjXyNQRgDdMbkLW2aXNYzcXOu8MOY9SBnCtDYgwClFooApQRAwz1N//cP//2vvvcKviM+327gfOvnH8PwlE4RQANM3x6K5KMuX+OMvyk1VxgDNJtTeBCg1EIRoJQO0N9/byQX5M9tC9HlnvavQHgOoG9HDSpAPwr4OX2oqgHUzkmAUmuLAKU0gF7vdr/6/p/+p48/VAD6oj/81ZWa78jwda5ALzd+Zl0/4+QVKNWNCFBKA+iwLekcfAYKGTfsIkrDJ7oNGbOfgcLknwW7mj673XTXAGrkVAZAgFILRYBSCkDnC7oX8AD+BL0ufBLh0134y1fZz/Iu/Pjd9hsiNYB+GNzqHxKrAAU5o/v2BCi1tghQKgvQj/BnoB8F+5a+8wkKH4lZtg8UATT+CvyH42cFGKBpzqmkz5QPcQlQaqEIUCrzFv7yvZ6RT+lHim/92eWrR7et92n4BVvhN5EuvP36Y/RNpEuzAtDwHfwt4yc6QNOc43ehfvtdBaBjSRTVKAKU0gD6j8P3xl/94DdXQgr+zDuMfgTC4+/CX+l3/QxT/S48AOhLMeFm/duhDtAk53Bf64Wiv8EAHUuiqEYRoJS6jen6y0WXvfLafZ6vf/v9y0bQH36Mws/RrzG96PcvB29/kv4a09AZA/TydfvwTfZHF1YbAI1zDiV968+0u/BjSRTVKAKUoiiqUQQoRVFUowhQiqKoRhGgFEVRjSJAKYqiGkWAUhRFNYoApSiKahQBSlEU1SgClKIoqlEEKEVRVKMIUIqiqEYRoBRFUY0iQCmKohpFgFIURTWKAKUoimoUAUpRFNUoApSiKKpRBChFUVSjCFCKoqhGEaAURVGN+v9mwnzENbyw8QAAAABJRU5ErkJggg=="
role="img" width="672" />

``` r
# ดูคาแรกเตอร์ของแต่ละกลุ่ม (Cluster)
cluster_characteristics <- data %>%
  group_by(Cluster) %>%
  summarise(
    Avg_TransactionAmount = mean(TransactionAmount, na.rm = TRUE),
    Avg_TransactionDuration = mean(TransactionDuration, na.rm = TRUE),
    Avg_LoginAttempts = mean(LoginAttempts, na.rm = TRUE),
    Avg_AccountBalance = mean(AccountBalance, na.rm = TRUE),
    Count = n()
  )
print(cluster_characteristics)
```

    ## # A tibble: 3 × 6
    ##   Cluster Avg_TransactionAmount Avg_TransactionDuration Avg_LoginAttempts
    ##     <int>                 <dbl>                   <dbl>             <dbl>
    ## 1       0                  599.                    136.              2.68
    ## 2       1                  272.                    118.              1   
    ## 3       2                  853.                    234.              1   
    ## # ℹ 2 more variables: Avg_AccountBalance <dbl>, Count <int>

``` r
# กรองเฉพาะกลุ่ม Outliers (Cluster 0)
anomalies <- data %>% filter(Cluster == 0)

# สรุปข้อมูลลักษณะสำคัญของ Outliers
anomalies_summary <- anomalies %>%
  summarise(
    Avg_Age = mean(CustomerAge, na.rm = TRUE),             # อายุเฉลี่ย
    Median_Age = median(CustomerAge, na.rm = TRUE),        # ค่ากลางของอายุ
    Common_Occupation = names(which.max(table(CustomerOccupation))), # อาชีพที่พบบ่อยที่สุด
    Common_Location = names(which.max(table(Location))),   # สถานที่ที่พบบ่อยที่สุด
    Common_Channel = names(which.max(table(Channel))),     # ช่องทางที่พบบ่อยที่สุด
    Avg_TransactionAmount = mean(TransactionAmount, na.rm = TRUE), # ค่าเฉลี่ยของมูลค่าธุรกรรม
    Median_TransactionAmount = median(TransactionAmount, na.rm = TRUE), # ค่ากลางของมูลค่าธุรกรรม
    Avg_TransactionDuration = mean(TransactionDuration, na.rm = TRUE),  # ระยะเวลาธุรกรรมเฉลี่ย
    Median_TransactionDuration = median(TransactionDuration, na.rm = TRUE), # ค่ากลางของระยะเวลาธุรกรรม
    Avg_AccountBalance = mean(AccountBalance, na.rm = TRUE), # ยอดเงินเฉลี่ยในบัญชี
    Median_AccountBalance = median(AccountBalance, na.rm = TRUE), # ค่ากลางของยอดเงินในบัญชี
    Count = n()  # จำนวนธุรกรรมทั้งหมดในกลุ่มนี้
  )

# แสดงผลลัพธ์
print(anomalies_summary)
```

    ##    Avg_Age Median_Age Common_Occupation Common_Location Common_Channel
    ## 1 45.19892         46            Doctor      Fort Worth            ATM
    ##   Avg_TransactionAmount Median_TransactionAmount Avg_TransactionDuration
    ## 1              598.8508                   399.73                136.4892
    ##   Median_TransactionDuration Avg_AccountBalance Median_AccountBalance Count
    ## 1                      129.5           6138.282               5621.15   186

``` r
# กรองเฉพาะ Outliers (Cluster = 0)
anomalies <- data %>% filter(Cluster == 0)

# หาค่าที่พบบ่อยที่สุด 5 อันดับในตัวแปรสำคัญ
top_occupations <- anomalies %>%
  count(CustomerOccupation, sort = TRUE) %>%
  slice_max(order_by = n, n = 5)

top_locations <- anomalies %>%
  count(Location, sort = TRUE) %>%
  slice_max(order_by = n, n = 5)

top_channels <- anomalies %>%
  count(Channel, sort = TRUE) %>%
  slice_max(order_by = n, n = 5)

# แสดงผลลัพธ์
cat("Top 5 Occupations in Cluster 0:\n")
```

    ## Top 5 Occupations in Cluster 0:

``` r
print(top_occupations)
```

    ##   CustomerOccupation  n
    ## 1             Doctor 58
    ## 2           Engineer 50
    ## 3            Retired 39
    ## 4            Student 39

``` r
cat("\nTop 5 Locations in Cluster 0:\n")
```

    ## 
    ## Top 5 Locations in Cluster 0:

``` r
print(top_locations)
```

    ##         Location n
    ## 1     Fort Worth 8
    ## 2  Oklahoma City 7
    ## 3   Philadelphia 7
    ## 4        Atlanta 6
    ## 5         Dallas 6
    ## 6   Jacksonville 6
    ## 7    Kansas City 6
    ## 8    Los Angeles 6
    ## 9        Memphis 6
    ## 10         Miami 6
    ## 11       Phoenix 6

``` r
cat("\nTop 5 Channels in Cluster 0:\n")
```

    ## 
    ## Top 5 Channels in Cluster 0:

``` r
print(top_channels)
```

    ##   Channel  n
    ## 1     ATM 65
    ## 2  Online 63
    ## 3  Branch 58

``` r
# นับจำนวนทั้งหมดในแต่ละตัวแปร (ข้อมูลทั้งหมด)
total_occupations <- data %>%
  count(CustomerOccupation, sort = TRUE) %>%
  rename(Total = n)

total_locations <- data %>%
  count(Location, sort = TRUE) %>%
  rename(Total = n)

total_channels <- data %>%
  count(Channel, sort = TRUE) %>%
  rename(Total = n)

# นับจำนวนใน Outliers (Cluster 0)
outlier_occupations <- data %>%
  filter(Cluster == 0) %>%
  count(CustomerOccupation, sort = TRUE) %>%
  rename(Outliers = n)

outlier_locations <- data %>%
  filter(Cluster == 0) %>%
  count(Location, sort = TRUE) %>%
  rename(Outliers = n)

outlier_channels <- data %>%
  filter(Cluster == 0) %>%
  count(Channel, sort = TRUE) %>%
  rename(Outliers = n)

# รวมตาราง Outliers กับข้อมูลทั้งหมด
comparison_occupations <- total_occupations %>%
  left_join(outlier_occupations, by = "CustomerOccupation") %>%
  mutate(Outliers_Percent = (Outliers / Total) * 100)

comparison_locations <- total_locations %>%
  left_join(outlier_locations, by = "Location") %>%
  mutate(Outliers_Percent = (Outliers / Total) * 100)

comparison_channels <- total_channels %>%
  left_join(outlier_channels, by = "Channel") %>%
  mutate(Outliers_Percent = (Outliers / Total) * 100)

# แสดงผลลัพธ์
cat("Comparison of Occupations:\n")
```

    ## Comparison of Occupations:

``` r
print(comparison_occupations)
```

    ##   CustomerOccupation Total Outliers Outliers_Percent
    ## 1            Student   657       39         5.936073
    ## 2             Doctor   631       58         9.191759
    ## 3           Engineer   625       50         8.000000
    ## 4            Retired   599       39         6.510851

``` r
cat("\nComparison of Locations:\n")
```

    ## 
    ## Comparison of Locations:

``` r
print(comparison_locations)
```

    ##            Location Total Outliers Outliers_Percent
    ## 1        Fort Worth    70        8        11.428571
    ## 2       Los Angeles    69        6         8.695652
    ## 3         Charlotte    68        4         5.882353
    ## 4     Oklahoma City    68        7        10.294118
    ## 5      Philadelphia    67        7        10.447761
    ## 6            Tucson    67        2         2.985075
    ## 7             Omaha    65        2         3.076923
    ## 8             Miami    64        6         9.375000
    ## 9           Detroit    63        5         7.936508
    ## 10          Houston    63        5         7.936508
    ## 11          Memphis    63        6         9.523810
    ## 12           Denver    62        1         1.612903
    ## 13          Atlanta    61        6         9.836066
    ## 14           Boston    61        4         6.557377
    ## 15      Kansas City    61        6         9.836066
    ## 16             Mesa    61        5         8.196721
    ## 17          Seattle    61        2         3.278689
    ## 18          Chicago    60        4         6.666667
    ## 19 Colorado Springs    60        5         8.333333
    ## 20           Fresno    60        2         3.333333
    ## 21     Jacksonville    60        6        10.000000
    ## 22           Austin    59        5         8.474576
    ## 23          Raleigh    59        2         3.389831
    ## 24      San Antonio    59        3         5.084746
    ## 25        San Diego    59        5         8.474576
    ## 26         San Jose    59        5         8.474576
    ## 27     Indianapolis    58        4         6.896552
    ## 28         New York    58        5         8.620690
    ## 29    San Francisco    57        2         3.508772
    ## 30        Las Vegas    55        5         9.090909
    ## 31        Milwaukee    55        5         9.090909
    ## 32        Nashville    55        3         5.454545
    ## 33          Phoenix    55        6        10.909091
    ## 34   Virginia Beach    55        5         9.090909
    ## 35         Columbus    54        5         9.259259
    ## 36       Sacramento    53        3         5.660377
    ## 37        Baltimore    51        1         1.960784
    ## 38       Louisville    51        3         5.882353
    ## 39           Dallas    49        6        12.244898
    ## 40       Washington    48        3         6.250000
    ## 41          El Paso    46        5        10.869565
    ## 42         Portland    42        2         4.761905
    ## 43      Albuquerque    41        4         9.756098

``` r
cat("\nComparison of Channels:\n")
```

    ## 
    ## Comparison of Channels:

``` r
print(comparison_channels)
```

    ##   Channel Total Outliers Outliers_Percent
    ## 1  Branch   868       58         6.682028
    ## 2     ATM   833       65         7.803121
    ## 3  Online   811       63         7.768187

</div>

</div>

</div>

</div>
