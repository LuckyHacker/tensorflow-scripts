import paramiko
import apache_log_parser
import pandas as pd
import numpy as np
import csv


host = "172.17.0.225"
csv_file_path = "data/access.csv"
log_file_path = "data/access.log"


def fetch_access_log():
    paramiko.util.log_to_file('/tmp/paramiko.log')

    port = 22
    transport = paramiko.Transport((host, port))

    username = "root"
    password = "root66"
    transport.connect(username=username, password=password)

    sftp = paramiko.SFTPClient.from_transport(transport)

    filepath = '/var/log/apache2/access.log'
    localpath = 'data/access.log'
    sftp.get(filepath, localpath)

    sftp.close()
    transport.close()


def parse_log():
    csv_header_fields = [   "date",
                            "time",
                            "request_header_user_agent__browser__version_string",
                            "request_header_referer",
                            "request_url",
                            "request_url_username",
                            "remote_host",
                            "request_header_user_agent__os__version_string",
                            "request_url_password",
                            "request_url_path",
                            "request_url_port",
                            "request_url_query",
                            "request_http_ver",
                            "response_bytes_clf",
                            "request_url_scheme",
                            "request_header_user_agent__is_mobile",
                            "request_url_netloc",
                            "request_header_user_agent__browser__family",
                            "request_url_hostname",
                            "remote_user",
                            "request_header_user_agent__os__family",
                            "request_method",
                            "request_first_line",
                            "request_url_fragment",
                            "status",
                            "request_header_user_agent",
                            "remote_logname"
                            ]

    parsed_lines = []
    line_parser = apache_log_parser.make_parser('%h %l %u %t \"%r\" %>s %b \"%{Referer}i\" \"%{User-Agent}i')
    with open(csv_file_path, 'w') as csvfile:
        csvfile.write(",".join(csv_header_fields) + "\n")
        with open(log_file_path, "r") as f:
            for line in f.readlines():
                parsed_line = line_parser(line)
                date, time = parsed_line["time_received_isoformat"].split("T")
                csv_line = [date,
                            time,
                            parsed_line.get("request_header_user_agent__browser__version_string", "None"),
                            parsed_line.get("request_header_referer", "None"),
                            parsed_line.get("request_url", "None"),
                            parsed_line.get("request_url_username", "None"),
                            parsed_line.get("remote_host", "None"),
                            parsed_line.get("request_header_user_agent__os__version_string", "None"),
                            parsed_line.get("request_url_password", "None"),
                            parsed_line.get("request_url_path", "None"),
                            parsed_line.get("request_url_port", "None"),
                            parsed_line.get("request_url_query", "None"),
                            parsed_line.get("request_http_ver", "None"),
                            parsed_line.get("response_bytes_clf", "None"),
                            parsed_line.get("request_url_scheme", "None"),
                            parsed_line.get("request_header_user_agent__is_mobile", "None"),
                            parsed_line.get("request_url_netloc", "None"),
                            parsed_line.get("request_header_user_agent__browser__family", "None"),
                            parsed_line.get("request_url_hostname", "None"),
                            parsed_line.get("remote_user", "None"),
                            parsed_line.get("request_header_user_agent__os__family", "None"),
                            parsed_line.get("request_method", "None"),
                            parsed_line.get("request_first_line", "None"),
                            parsed_line.get("request_url_fragment", "None"),
                            parsed_line.get("status", "None"),
                            parsed_line.get("request_header_user_agent", "None"),
                            parsed_line.get("remote_logname", "None"),
                            ]

                writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                writer.writerow(string_to_scalar(csv_line))


def string_to_scalar(string_list):
    scalar_list = []
    for s in string_list:
        if s != "":
            scalar_list.append(float(sum(list(map(lambda x: ord(x), str(s)))) / len(str(s))))
        else:
            scalar_list.append(float(0))

    return scalar_list



def fill_csv_null():
    df = pd.read_csv(csv_file_path)
    df = df.replace(np.NaN, "None", regex=True)
    df.to_csv(csv_file_path, index=False)



if __name__ == "__main__":
    #fetch_access_log()
    parse_log()
    fill_csv_null()
