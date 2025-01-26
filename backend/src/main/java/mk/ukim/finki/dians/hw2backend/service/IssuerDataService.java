package mk.ukim.finki.dians.hw2backend.service;

import mk.ukim.finki.dians.hw2backend.model.IssuerData;

import java.util.List;

public interface IssuerDataService {

    public List<IssuerData> getAllData();

    public List<IssuerData> getDataByIssuer(String issuer);

    public List<String> getAllIssuers();
}
