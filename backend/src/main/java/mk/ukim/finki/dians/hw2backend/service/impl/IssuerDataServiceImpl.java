package mk.ukim.finki.dians.hw2backend.service.impl;

import mk.ukim.finki.dians.hw2backend.model.IssuerData;
import mk.ukim.finki.dians.hw2backend.repository.IssuerDataRepository;
import mk.ukim.finki.dians.hw2backend.service.IssuerDataService;
import org.springframework.stereotype.Service;

import java.util.List;

@Service
public class IssuerDataServiceImpl implements IssuerDataService {

    final IssuerDataRepository issuerDataRepository;

    public IssuerDataServiceImpl(IssuerDataRepository issuerDataRepository) {
        this.issuerDataRepository = issuerDataRepository;
    }

    @Override
    public List<IssuerData> getAllData() {
        return issuerDataRepository.findAll();
    }

    @Override
    public List<IssuerData> getDataByIssuer(String issuer) {
        return issuerDataRepository.findByIssuer(issuer);
    }

    public List<String> getAllIssuers() {
        return issuerDataRepository.getAllIssuers();
    }
}
