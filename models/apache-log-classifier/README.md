# Apache log classifier

## Results

#### Nikto scan

Detecting Nikto is relatively easy, because it is not silent by any means. Classifying log entries (access.log) when scanned with Nikto, model scored 95% accuracy:

```
Accuracy: 95.0%
172.17.0.225 - - [23/Oct/2017:11:11:06 +0300] "POST /wp-cron.php?doing_wp_cron=1508746266.3259329795837402343750 HTTP/1.1" 200 166 "-" "WordPress/4.8.2; http://172.17.0.225"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.html~ HTTP/1.1" 404 506 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:28:24 +0300] "GET /wp-includes/js/wp-emoji-release.min.js?ver=4.8.2 HTTP/1.1" 200 4674 "http://172.17.0.225/" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:05:26 +0300] "GET /wp-admin/js/language-chooser.min.js?ver=4.8.2 HTTP/1.1" 200 590 "http://172.17.0.225/wp-admin/setup-config.php" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.passwd HTTP/1.1" 404 507 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.php+ HTTP/1.1" 404 505 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:09:49 +0300] "GET /wp-includes/js/underscore.min.js?ver=1.8.3 HTTP/1.1" 200 6173 "http://172.17.0.225/wp-admin/install.php?language=en_US" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:11:12 +0300] "GET /wp-admin/load-scripts.php?c=0&load%5B%5D=jquery-core,jquery-migrate,utils&ver=4.8.2 HTTP/1.1" 200 38174 "http://172.17.0.225/wp-admin/" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:09:11 +0300] "POST /wp-admin/setup-config.php?step=2 HTTP/1.1" 200 2767 "http://172.17.0.225/wp-admin/setup-config.php?step=1" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
::1 - - [23/Oct/2017:11:11:22 +0300] "OPTIONS * HTTP/1.0" 200 126 "-" "Apache/2.4.18 (Ubuntu) (internal dummy connection)"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.idq HTTP/1.1" 404 504 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:15:43 +0300] "POST /wp-admin/admin-ajax.php HTTP/1.1" 200 435 "http://172.17.0.225/wp-admin/edit.php" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
192.168.1.6 - - [13/Oct/2017:16:33:05 +0300] "GET /favicon.ico HTTP/1.1" 404 502 "http://192.168.1.3/" "Mozilla/5.0 (Linux; Android 7.1.1; F5121 Build/34.3.A.0.228) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.98 Mobile Safari/537.36"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.rdf+destype=cache+desformat=PDF HTTP/1.1" 404 532 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:10:53:43 +0300] "GET / HTTP/1.1" 200 3525 "-" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.de HTTP/1.1" 404 503 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
172.17.8.62 - - [23/Oct/2017:11:08:48 +0300] "GET /wp-admin/setup-config.php?step=1 HTTP/1.1" 200 1388 "http://172.17.0.225/wp-admin/setup-config.php" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:14:27 +0300] "POST /wp-admin/admin-ajax.php HTTP/1.1" 200 436 "http://172.17.0.225/wp-admin/" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:11:27 +0300] "GET /wp-admin/admin-ajax.php?action=wp-compression-test&test=no&_ajax_nonce=8ab3475170&1508746290183 HTTP/1.1" 200 440 "http://172.17.0.225/wp-admin/" "Mozilla/5.0 (X11; Linux x86_64; rv:57.0) Gecko/20100101 Firefox/57.0"    pred: 0
172.17.8.62 - - [23/Oct/2017:11:37:13 +0300] "GET /GALYi6Qt.idc HTTP/1.1" 404 504 "-" "Mozilla/5.00 (Nikto/2.1.5) (Evasions:None) (Test:map_codes)"    pred: 1
```
