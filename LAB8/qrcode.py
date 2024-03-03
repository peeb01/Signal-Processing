link = "https://github.com/peeb01/Signal-Processing/tree/main/LAB8"

import segno

qrcode = segno.make_qr(link)
qrcode.save("qrcode.png", scale=15)