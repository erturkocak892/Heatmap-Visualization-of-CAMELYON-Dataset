# CAMELYON17 Data Set
## Overview
Built on the success of its predecessor, CAMELYON17 is the second grand challenge in pathology organised by the [Computational Pathology Group](http://www.diagnijmegen.nl/index.php/Digital_Pathology) of the Radboud University Medical Center (Radboudumc) in Nijmegen, The Netherlands.


The goal of this challenge is to evaluate new and existing algorithms for automated detection and classification of breast cancer metastases in whole-slide images of histological lymph node sections. This task has high clinical relevance and would normally require extensive microscopic assessment by pathologists. The presence of metastases in lymph nodes has therapeutic implications for breast cancer patients. Therefore, an automated solution would hold great promise to reduce the workload of pathologists while at the same time reduce the subjectivity in diagnosis.


For the complete description of the challenge and the data set please visit the [challenge](https://camelyon17.grand-challenge.org) website.

## Data
### Images
The data in this challenge contains a total of 1000 whole-slide images (WSIs) of sentinel lymph node from 5 different medical centers from The Netherlands: Radboud University Medical Center in Nijmegen, Canisius-Wilhelmina Hospital in Nijmegen, University Medical Center Utrecht, Rijnstate Hospital in Arnhem, and Laboratorium Pathologie Oost-Nederland in Hengelo.

The data set is divided into training and testing sets with 20 patients from each center in both sets. For each patient the shared 5 whole-slide images are zipped together into a single ZIP file. The patient pN-stages and the slide-level labels in the training set are shared in the *stage_labels.csv* file.

The slides are converted to generic [TIFF](https://www.awaresystems.be/imaging/tiff/bigtiff.html) (Tagged Image File Format) using an open-source file converter, part of the [ASAP](https://github.com/GeertLitjens/ASAP) package.

### Annotations
From each center 10 slides are exhaustively annotated and the annotations are shared in XML format. The XML files are compatible with the [ASAP](https://github.com/GeertLitjens/ASAP) software. You may download this software and visualize the annotations overlaid on the whole slide image.

The provided XML files may have two groups of annotations ("metastases", or "normal") which can be accessed from the "PartOfGroup" attribute of the Annotation node in the XML file. Annotations belonging to group "metastases" represent tumor areas and annotations within group "normal" are non-tumor areas which have been cut-out from the original annotations in the "metastases" group.

### Integrity
The *checksums.md5* file contains the MD5 checksums of all the shared CAMELYON17 files. The downloaded files can be checked against this list with *md5sum*.

### Licensing
See *license.txt* for licensing information.
