import paramiko
import apache_log_parser
import pandas as pd
import numpy as np


host = "192.168.1.5"
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
    with open("data/access.csv", "w") as csv:
        csv.write(",".join(csv_header_fields) + "\n")
        with open("data/access.log", "r") as f:
            for line in f.readlines():
                parsed_line = line_parser(line)
                date, time = parsed_line["time_received_isoformat"].split("T")
                csv_line = "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                        date,
                        time,
                        parsed_line["request_header_user_agent__browser__version_string"],
                        parsed_line["request_header_referer"],
                        parsed_line["request_url"],
                        parsed_line["request_url_username"],
                        parsed_line["remote_host"],
                        parsed_line["request_header_user_agent__os__version_string"],
                        parsed_line["request_url_password"],
                        parsed_line["request_url_path"],
                        parsed_line["request_url_port"],
                        parsed_line["request_url_query"],
                        parsed_line["request_http_ver"],
                        parsed_line["response_bytes_clf"],
                        parsed_line["request_url_scheme"],
                        parsed_line["request_header_user_agent__is_mobile"],
                        parsed_line["request_url_netloc"],
                        parsed_line["request_header_user_agent__browser__family"],
                        parsed_line["request_url_hostname"],
                        parsed_line["remote_user"],
                        parsed_line["request_header_user_agent__os__family"],
                        parsed_line["request_method"],
                        parsed_line["request_first_line"],
                        parsed_line["request_url_fragment"],
                        parsed_line["status"],
                        parsed_line["request_header_user_agent"],
                        parsed_line["remote_logname"])

                csv.write(csv_line)


def fill_csv_null():
    df = pd.read_csv(csv_file_path)
    df = df.replace(np.NaN, "None", regex=True)
    df.to_csv(csv_file_path)



if __name__ == "__main__":
    #fetch_access_log()
    parse_log()
    fill_csv_null()
