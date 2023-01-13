#****************************************************************************************
# Modules                                                                               *
#****************************************************************************************
import h5py as h5
#****************************************************************************************



#****************************************************************************************
# Definition                                                                            *
#****************************************************************************************
def HDF5load(hdf5_file):
	info = h5.File(hdf5_file,'r')
	data = dict.fromkeys([])
	#Datasets
	for key in info.keys():
		keyDetail = info[key]
		isDataSet1=isinstance(keyDetail,h5.Dataset)
		if isDataSet1:
			shape1 = keyDetail.shape[0]	
			data[key] = keyDetail[0:shape1]
		else :
			for group in keyDetail.keys():
				groupMemberDetail = keyDetail[group]
				isDataSet2 = isinstance(groupMemberDetail,h5.Dataset)
				if isDataSet2:
					shape2 = groupMemberDetail.shape[0]
					data[group] = groupMemberDetail[0:shape2]
				else :
					for subGroup in groupMemberDetail.keys():
						subGroupMemberDetail = groupMemberDetail[subGroup]
						shape3 = subGroupMemberDetail.shape[0]
						data[subGroup] = subGroupMemberDetail[0:shape3]				
	return data
#****************************************************************************************	

		
						
						
				
			
			

    			
    			
			

