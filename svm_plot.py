from matplotlib import pyplot as plt

# svm_fea20
# max_c:  10  max_gamma:  0.02  max_score:  0.6782050391078798
matrix = [[0.6778563708792864, 0.6778563708792864, 0.6778563708792864, 0.6778563708792864, 0.6776240828421203, 0.6777403449373248, 0.6770434470924491, 0.6766366731113863],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241671951648, 0.6776241671951648, 0.678030772479939, 0.6774500186340453, 0.6776243358914538],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6777403449569259, 0.678030772479939, 0.6773919297580651, 0.6776243358914538],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241672049654, 0.6778565058421977, 0.6775661457859395, 0.6776243358914538],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241672049654, 0.6778565058421977, 0.6777404124040799, 0.6776243358914538],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241672049654, 0.6777403280902373, 0.6782050391078798, 0.6776243358914538],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241672049654, 0.6777403280902373, 0.6782050391078798, 0.6774499848908673],
          [0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6775079894432043, 0.6776241672049654, 0.6777403280902373, 0.6780307724701384, 0.6772757013962377]]

plt.title("20 features svm parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.68, vmin=0.677)

plt.xlabel("C")
plt.ylabel("sigma")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()

# lle_svm_fea20
# max_c:  50  max_gamma:  5000  max_sigma:  0.01  max_score:  0.6358619728938056
matrix = [[0.6297628939338349, 0.6082130477563242, 0.5613963623825154, 0.5537871784582801, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6311569864773104, 0.6112915220155458, 0.6082711298844534, 0.5537871784582801, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6351649468667971, 0.6108849774857354, 0.6140217666874721, 0.5537871784582801, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6351649468667971, 0.6108849774857354, 0.6299950604885178, 0.5584921243853169, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6351649468667971, 0.6202948794621737, 0.6086195922861346, 0.5926463086074536, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6351649468667971, 0.6202948794621737, 0.6093747510530955, 0.6020563421746317, 0.5616287313844907,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6351649468667971, 0.6201787152059154, 0.6108849774857354, 0.6086777047813576, 0.5926463086074536,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.6358619728938056, 0.6243027993622019, 0.6108849774857354, 0.6181454661939036, 0.6023467528152776,
           0.5926463086074536, 0.5537871784582801, 0.5537871784582801]]

plt.title("20 features lle_svm parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.65, vmin=0.55)

plt.xlabel("C")
plt.ylabel("sigma")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()

# isomap_svm_fea20
# max_c:  50  max_gamma:  5000  max_sigma:  0.01  max_score:  0.5719099766288772  std:  0.006811444809206055
matrix = [[0.569993175299333, 0.5537871784582801, 0.5537871784582801, 0.5537871784582801, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.5703997501962372, 0.5659271833935402, 0.55378717845828, 0.5537871784582801, 0.5537871784582801,
           0.5537871784582801, 0.5537871784582801, 0.5537871784582801],
          [0.571212970846598, 0.565869101265411, 0.5604088346135797, 0.5537290963301508, 0.5537290963301508,
           0.5537290963301508, 0.5537871784582801, 0.5537871784582801],
          [0.571212970846598, 0.565869101265411, 0.5658110191372817, 0.5537290963301508, 0.5537290963301508,
           0.5537290963301508, 0.5537290963301508, 0.5537871784582801],
          [0.5713291553475854, 0.5658110191372817, 0.5658110191372817, 0.5536710142020217, 0.5537290963301508,
           0.5537290963301508, 0.5537290963301508, 0.5537871784582801],
          [0.5713291553475854, 0.5657529370091526, 0.5658110191372817, 0.5536710142020217, 0.5537290963301508,
           0.5537290963301508, 0.5537290963301508, 0.5537290963301508],
          [0.5713291553475854, 0.5658110191372817, 0.5658110191372817, 0.5536129320738925, 0.5536710142020217,
           0.5537290963301508, 0.5537290963301508, 0.5537290963301508],
          [0.5719099766288772, 0.5699931854216976, 0.5657529370091526, 0.5536129320738925, 0.5536710142020217,
           0.5537290963301508, 0.5536710142020217, 0.5536710142020217]]

plt.title("20 features isomap_svm parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.58, vmin=0.55)

plt.xlabel("C")
plt.ylabel("sigma")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()

# pca_svm_fea20
# max_c:  0.5  max_gamma:  199.99999999999997 max_score: 0.6778574164935792
matrix = [[0.6777412421149563, 0.677683159986827, 0.677683159986827, 0.6723392904056403, 0.6507898592450785, 0.638301786680357, 0.615764473468098, 0.6159386996077564],
          [0.677741252237321, 0.6778574164935792, 0.6778574164935792, 0.6777993242430855, 0.6528228450756104, 0.653461728240302, 0.6135572614979079, 0.615764473468098],
          [0.677741252237321, 0.6778574164935792, 0.6778574164935792, 0.6777993242430855, 0.6619424376350488, 0.6517772048114514, 0.6159970449173654, 0.615764473468098],
          [0.677741252237321, 0.6778574164935792, 0.6778574164935792, 0.6776831498644625, 0.6698996891887458, 0.6595606148753452, 0.6211664150550495, 0.6143122987964933],
          [0.677741252237321, 0.677741252237321, 0.6778574164935792, 0.6776250677363334, 0.6731524705665425, 0.6633360746721169, 0.625813106773759, 0.6136153537484018],
          [0.677741252237321, 0.677741252237321, 0.6778574164935792, 0.6776250677363334, 0.677392627877806, 0.6570051227060368, 0.6253484497487255, 0.6134991692474142],
          [0.677741252237321, 0.677741252237321, 0.6778574164935792, 0.6776250677363334, 0.6777993242430855, 0.659734932116285, 0.6322608910721614, 0.6144285946434916],
          [0.677741252237321, 0.677741252237321, 0.6776831701091918, 0.6776250677363334, 0.6777993242430855, 0.6635687271003011, 0.6322028393111262, 0.6211664150550495]]

plt.title("20 features pca_svm parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.68, vmin=0.6)

plt.xlabel("C")
plt.ylabel("sigma")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()

# tsne_svm_fea20
# max_c:  0.5  max_gamma:  0.005 max_score: 0.6783801657691065
matrix = [[0.6668217311701192, 0.6669379055487422, 0.6669959876768714, 0.6669959876768714, 0.6768119280648895, 0.6768119280648895, 0.6781478170118606, 0.6780897348837313],
          [0.670538845657282, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6761148109365994, 0.6780897551284606, 0.6780897551284606, 0.6783801657691065],
          [0.670538845657282, 0.6706550200359048, 0.6706550200359048, 0.6707130920416695, 0.6760567288084705, 0.6780897551284606, 0.6780897551284606, 0.6780897551284606],
          [0.670538845657282, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6744885012266181, 0.6780897551284606, 0.6780897551284606, 0.6776831802315564],
          [0.670538845657282, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6744885012266181, 0.6771602791205601, 0.6780897551284606, 0.6776831802315564],
          [0.670538845657282, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6744885012266181, 0.6755920616610723, 0.6780897551284606, 0.6776831802315564],
          [0.6704226814010236, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6744304190984888, 0.6755920616610723, 0.6780316730003314, 0.6776831802315564],
          [0.6704226814010236, 0.6706550200359048, 0.6706550200359048, 0.6706550099135402, 0.6744304190984888, 0.6755920616610723, 0.6753018028558957, 0.6776831802315564]]

plt.title("20 features tsne_svm parameters vs accuracy scores")
plt.imshow(matrix, cmap="hot", vmax=0.68, vmin=0.66)

plt.xlabel("C")
plt.ylabel("sigma")
ax = plt.colorbar()
ax.set_label("accuracy score")
plt.show()
