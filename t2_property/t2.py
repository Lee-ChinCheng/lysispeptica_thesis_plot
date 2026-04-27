import csv

### input & output setting
input_txt = 't2_property/t2_property.txt'
output_csv = 't2_property/t2_property.csv'
#========================================


#Sequence:	condifential
#Length:	19
#Mass:	1940.1536
#Isoelectric point (pI):	10.64
#Net charge:	+2
#Hydrophobicity:	+16.78 Kcal * mol -1



def clean_value(line):
    """Extract value after colon and remove units/symbols."""
    value = line.split(":", 1)[1].strip()
    # Remove units
    value = value.replace("Kcal * mol -1", "").strip()
    # Remove '+' sign from numbers
    value = value.replace("+", "")
    
    return value

with open(input_txt, "r") as f:
    lines = [line.strip() for line in f if line.strip()]



# Every 6 lines correspond to one peptide
num_properties = 6
rows = []
print(len(lines))

id_num=0
for i in range(0, len(lines), num_properties):
    block = lines[ i :i + num_properties]

    if len(block) == num_properties:
        id_num+=1
        row = [clean_value(line) for line in block]
        row[0] = "confidential"
        #row[3] = round( float(row[3]) ,2)

        row.insert(0,f'AMP{id_num}')
        rows.append(row)

with open(output_csv, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["ID","Seq", "Length", "Mass (amu)", "Isoelectric point ", "Net_charge", "Hydrophobicity (Kcal * mol -1)"])
    writer.writerows(rows)


